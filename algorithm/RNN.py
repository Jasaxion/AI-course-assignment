import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm

class Vocabulary:
    def __init__(self):
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_char = {0: '<PAD>', 1: '<UNK>'}
        self.tag_to_idx = {'<PAD>': 0, 'B': 1, 'M': 2, 'E': 3, 'S': 4}
        self.idx_to_tag = {0: '<PAD>', 1: 'B', 2: 'M', 3: 'E', 4: 'S'}
        
    def add_character(self, char: str) -> None:
        if char not in self.char_to_idx:
            idx = len(self.char_to_idx)
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
            
    def get_vocab_size(self) -> int:
        return len(self.char_to_idx)
    
    def get_tag_size(self) -> int:
        return len(self.tag_to_idx)

class ChineseSegmentationDataset(Dataset):
    def __init__(self, data_path: str, vocab: Vocabulary, max_len: int = 50, is_test: bool = False):
        self.vocab = vocab
        self.max_len = max_len
        self.is_test = is_test
        self.sentences, self.labels = self.load_data(data_path)

    def load_data(self, data_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        sentences = []
        labels = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                if not words:
                    continue
                    
                sentence = []
                label = []
                for word in words:
                    if not self.is_test:  # Only add characters to vocab during training
                        for char in word:
                            self.vocab.add_character(char)
                        
                    if len(word) == 1:
                        sentence.append(word)
                        label.append('S')
                    else:
                        sentence.extend(list(word))
                        label.extend(['B'] + ['M'] * (len(word)-2) + ['E'])
                
                if len(sentence) > self.max_len:
                    sentence = sentence[:self.max_len]
                    label = label[:self.max_len]
                    
                sentences.append(sentence)
                labels.append(label)
        return sentences, labels

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sentence = self.sentences[idx]
        label = self.labels[idx]
        
        encoded_sentence = [self.vocab.char_to_idx.get(char, self.vocab.char_to_idx['<UNK>']) 
                          for char in sentence]
        encoded_label = [self.vocab.tag_to_idx[tag] for tag in label]
        
        return torch.tensor(encoded_sentence), torch.tensor(encoded_label)

class RNNSegmentation(nn.Module):
    def __init__(self, vocab_size: int, tag_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super(RNNSegmentation, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, 
                         num_layers=2,
                         batch_first=True,
                         dropout=0.3 if hidden_dim > 1 else 0)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_dim, tag_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        rnn_out = self.dropout(rnn_out)
        logits = self.linear(rnn_out)
        return logits

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    sentences, labels = zip(*batch)
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return sentences, labels

def evaluate_model(model: nn.Module, 
                  test_loader: DataLoader, 
                  vocab: Vocabulary,
                  device: str = 'cuda') -> Dict[str, float]:
    """
    评估模型性能，计算准确率、召回率和F1分数
    """
    model.eval()
    total_correct = 0
    total_pred = 0
    total_gold = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=-1)
            
            # 只考虑非填充位置的预测
            valid_mask = (targets != vocab.tag_to_idx['<PAD>'])
            pred_tags = predictions[valid_mask]
            gold_tags = targets[valid_mask]
            
            # 统计正确预测的边界标记数量
            correct_pred = (pred_tags == gold_tags).sum().item()
            total_correct += correct_pred
            total_pred += len(pred_tags)
            total_gold += len(gold_tags)
    
    # 计算评估指标
    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                test_loader: Optional[DataLoader] = None,
                vocab: Optional[Vocabulary] = None,
                epochs: int = 5, 
                device: str = 'cuda') -> Dict[str, float]:
    model.to(device)
    best_metrics = {'f1': 0.0}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        print(f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
                
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
        
        # Evaluation phase
        if test_loader is not None and vocab is not None:
            print("\nEvaluating on test set...")
            metrics = evaluate_model(model, test_loader, vocab, device)
            print(f"Test Metrics:")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}\n")
            
            # Save best model
            if metrics['f1'] > best_metrics['f1']:
                best_metrics = metrics
                print(f"New best F1 score: {metrics['f1']:.4f}, saving model...")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                }, './cache/best_rnn_segmentation_model.pt')
    
    return best_metrics

def segment_text(model: nn.Module, 
                text: str, 
                vocab: Vocabulary, 
                device: str = 'cuda') -> List[str]:
    model.eval()
    
    chars = list(text)
    char_indices = [vocab.char_to_idx.get(char, vocab.char_to_idx['<UNK>']) 
                   for char in chars]
    
    tensor_input = torch.tensor(char_indices).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor_input)
        predicted_tags = torch.argmax(outputs, dim=2)
    
    predicted_tags = [vocab.idx_to_tag[idx.item()] 
                     for idx in predicted_tags[0]]
    
    result = []
    current_word = chars[0]
    
    for i in range(1, len(chars)):
        if predicted_tags[i-1] in ['S', 'E']:
            result.append(current_word)
            current_word = chars[i]
        else:
            current_word += chars[i]
    
    if current_word:
        result.append(current_word)
    
    return result

def main():
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 设置数据路径
    base_path = '../dataset/icwb2-data'  # 修改为实际路径
    train_path = os.path.join(base_path, 'training/msr_training.utf8')
    test_path = os.path.join(base_path, 'testing/msr_test.utf8')
    gold_path = os.path.join(base_path, 'gold/msr_test_gold.utf8')
    
    # 初始化词表
    vocab = Vocabulary()
    
    # 加载训练集和测试集
    train_dataset = ChineseSegmentationDataset(train_path, vocab, max_len=50)
    test_dataset = ChineseSegmentationDataset(test_path, vocab, max_len=50, is_test=True)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, 
                            batch_size=32, 
                            shuffle=True, 
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,
                           batch_size=32,
                           shuffle=False,
                           collate_fn=collate_fn)
    
    # 初始化模型
    model = RNNSegmentation(vocab.get_vocab_size(), 
                          vocab.get_tag_size(),
                          embedding_dim=128,
                          hidden_dim=256)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD的loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练和评估模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_metrics = train_model(model, train_loader, criterion, optimizer, 
                             test_loader=test_loader, vocab=vocab,
                             epochs=10, device=device)
    
    print("\nBest model metrics:")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'optimizer_state_dict': optimizer.state_dict(),
    }, './model/RNN-based/rnn_segmentation_model.pt')

if __name__ == "__main__":
    main()