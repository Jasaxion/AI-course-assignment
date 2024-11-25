import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm

# 构建词表和标签映射
class Vocabulary:
    def __init__(self):
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_char = {0: '<PAD>', 1: '<UNK>'}
        self.tag_to_idx = {'<PAD>': 0, 'B': 1, 'M': 2, 'E': 3, 'S': 4}
        self.idx_to_tag = {0: '<PAD>', 1: 'B', 2: 'M', 3: 'E', 4: 'S'}
        
    def add_character(self, char):
        if char not in self.char_to_idx:
            idx = len(self.char_to_idx)
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
            
    def get_vocab_size(self):
        return len(self.char_to_idx)
    
    def get_tag_size(self):
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

class BiLSTMSegmentation(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim=128, hidden_dim=256):
        super(BiLSTMSegmentation, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                           num_layers=2, 
                           bidirectional=True, 
                           batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_dim, tag_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        logits = self.linear(lstm_out)
        return logits

def collate_fn(batch):
    sentences, labels = zip(*batch)
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return sentences, labels

def evaluate_model(model: nn.Module, 
                  data_loader: DataLoader, 
                  vocab: Vocabulary,
                  device: str = 'cuda') -> Dict[str, float]:
    """评估模型性能"""
    model.eval()
    total_correct = 0
    total_pred = 0
    total_gold = 0
    total_sequences = 0
    correct_sequences = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=-1)
            
            # 只考虑非填充位置的预测
            valid_mask = (targets != vocab.tag_to_idx['<PAD>'])
            pred_tags = predictions[valid_mask]
            gold_tags = targets[valid_mask]
            
            # 计算字符级别的指标
            correct_pred = (pred_tags == gold_tags).sum().item()
            total_correct += correct_pred
            total_pred += len(pred_tags)
            total_gold += len(gold_tags)
            
            # 计算序列级别的准确率
            for pred, gold, mask in zip(predictions, targets, valid_mask):
                total_sequences += 1
                seq_len = mask.sum().item()
                if torch.all(pred[:seq_len] == gold[:seq_len]):
                    correct_sequences += 1
    
    # 计算评估指标
    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sequence_accuracy': sequence_accuracy,
        'total_sequences': total_sequences
    }

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer,
                val_loader: Optional[DataLoader] = None,
                vocab: Optional[Vocabulary] = None,
                epochs: int = 5, 
                device: str = 'cuda',
                save_dir: str = './model',
                patience: int = 3) -> Dict[str, float]:
    """训练模型并返回最佳性能指标"""
    
    model.to(device)
    best_metrics = {'f1': 0.0}
    patience_counter = 0
    os.makedirs(save_dir, exist_ok=True)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                    factor=0.5, patience=2, verbose=True)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        print(f'\nEpoch {epoch+1}/{epochs}')
        
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, (inputs, targets) in enumerate(train_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch+1} - Average Training Loss: {avg_loss:.4f}')
        
        # Validation phase
        if val_loader is not None and vocab is not None:
            print("\nValidating...")
            metrics = evaluate_model(model, val_loader, vocab, device)
            
            print(f"Validation Metrics:")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Sequence Accuracy: {metrics['sequence_accuracy']:.4f}")
            
            # 学习率调整
            scheduler.step(metrics['f1'])
            
            # Early stopping check
            if metrics['f1'] > best_metrics['f1']:
                best_metrics = metrics
                patience_counter = 0
                print(f"New best F1 score: {metrics['f1']:.4f}, saving model...")
                torch.save({
                    'epoch': epoch + 1,
                    'vocab': vocab,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': metrics,
                }, os.path.join(save_dir, 'bilstm_segmentation_model.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
    
    return best_metrics

def segment_text(model, text, vocab, device='cuda'):
    model.eval()
    
    # 将文本转换为索引
    chars = list(text)
    char_indices = [vocab.char_to_idx.get(char, vocab.char_to_idx['<UNK>']) 
                   for char in chars]
    
    # 转换为tensor并移到设备
    tensor_input = torch.tensor(char_indices).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor_input)
        predicted_tags = torch.argmax(outputs, dim=2)
    
    # 将预测的标签转换回标签字符
    predicted_tags = [vocab.idx_to_tag[idx.item()] 
                     for idx in predicted_tags[0]]
    
    # 根据标签构建分词结果
    result = []
    current_word = chars[0]
    
    for i in range(1, len(chars)):
        if predicted_tags[i-1] in ['S', 'E']:
            result.append(current_word)
            current_word = chars[i]
        else:
            current_word += chars[i]
    
    # 添加最后一个词
    if current_word:
        result.append(current_word)
    
    return result

def main():
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # 设置数据路径
    base_path = '/home/shaoxiong/exa/人工智能大作业/dataset/icwb2-data'
    train_path = os.path.join(base_path, 'training/msr_training.utf8')
    test_path = os.path.join(base_path, 'testing/msr_test.utf8')
    gold_path = os.path.join(base_path, 'gold/msr_test_gold.utf8')
    
    # 初始化词表和数据集
    vocab = Vocabulary()
    train_dataset = ChineseSegmentationDataset(train_path, vocab, max_len=100)  # 增加最大长度
    test_dataset = ChineseSegmentationDataset(test_path, vocab, max_len=100, is_test=True)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, 
                            batch_size=64,  # 增大batch size
                            shuffle=True, 
                            collate_fn=collate_fn,
                            num_workers=4)  # 使用多进程加载数据
    
    test_loader = DataLoader(test_dataset,
                           batch_size=128,  # 测试时使用更大的batch
                           shuffle=False, 
                           collate_fn=collate_fn,
                           num_workers=4)
    
    # 初始化模型
    model = BiLSTMSegmentation(vocab.get_vocab_size(), 
                             vocab.get_tag_size(),
                             embedding_dim=256,  # 增大embedding维度
                             hidden_dim=512)  # 增大隐藏层维度
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 训练和评估
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = './model/BiLSTM-based'
    os.makedirs(save_dir, exist_ok=True)
    
    print("Starting training...")
    best_metrics = train_model(model, train_loader, criterion, optimizer,
                             val_loader=test_loader, vocab=vocab,
                             epochs=10, device=device, save_dir=save_dir)
    
    # 加载最佳模型进行最终测试
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(save_dir, 'bilstm_segmentation_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nEvaluating on test set...")
    final_metrics = evaluate_model(model, test_loader, vocab, device)
    
    print("\nFinal Test Metrics:")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print(f"Sequence Accuracy: {final_metrics['sequence_accuracy']:.4f}")
    print(f"Total Sequences: {final_metrics['total_sequences']}")


if __name__ == "__main__":
    main()