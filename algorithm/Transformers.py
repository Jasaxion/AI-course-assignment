import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm

#实现 Transformer 模型的位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #采用绝对位置编码方式
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, embedding_dim]
        return x

#完成多头注意力机制的实现
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        #定义Q、K、V的维度
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        #定义Q、K、V的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        #定义dropout
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.squeeze(1)  # Remove the extra dimension
        
        output, _ = self.attention(Q, K, V, key_padding_mask=mask)
        return self.dropout(output)
#实现Transformer的编码器层
#中文分词任务我们只需要采用编码器即可实现
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
#中文分词的Transformers模型
class ChineseSegmentationTransformer(nn.Module):
    def __init__(self, vocab_size: int, tag_size: int, d_model: int = 256, 
                 num_heads: int = 8, num_layers: int = 6, d_ff: int = 1024, 
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        
        # 缩小embedding维度，然后投影到d_model维度
        self.embedding_dim = d_model // 2
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        self.embed_proj = nn.Linear(self.embedding_dim, d_model)
        
        # Layer Norm在Embedding之后
        self.embed_norm = nn.LayerNorm(d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN architecture
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, tag_size)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # 特别处理embedding层
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        nn.init.zeros_(self.embedding.weight[0])  # padding idx
        
        # 输出层使用较小的初始值
        nn.init.normal_(self.output_layer.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 创建padding mask
        padding_mask = (x == 0)
        
        # Embedding and position encoding
        x = self.embedding(x)
        x = self.embed_proj(x)
        x = self.embed_norm(x)
        x = self.positional_encoding(x)
        
        # Transformer encoding
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        
        # Output
        x = self.output_dropout(x)
        x = self.output_norm(x)
        x = self.output_layer(x)
        
        return x
#定义词表类
class Vocabulary:
    def __init__(self):
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_char = {0: '<PAD>', 1: '<UNK>'}
        self.tag_to_idx = {'<PAD>': 0, 'B': 1, 'M': 2, 'E': 3, 'S': 4}
        self.idx_to_tag = {0: '<PAD>', 1: 'B', 2: 'M', 3: 'E', 4: 'S'}
        
    def add_character(self, char: str):
        if char not in self.char_to_idx:
            idx = len(self.char_to_idx)
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
            
    def get_vocab_size(self) -> int:
        return len(self.char_to_idx)
    
    def get_tag_size(self) -> int:
        return len(self.tag_to_idx)
#定义数据集类
class ChineseSegmentationDataset(Dataset):
    def __init__(self, data_path: str, vocab: Vocabulary, max_len: int = 512, is_test: bool = False):
        self.vocab = vocab
        self.max_len = max_len
        self.is_test = is_test
        self.sentences, self.labels = self.load_data(data_path)

    def load_data(self, data_path: str) -> tuple:
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
                    if not self.is_test:  # 仅在训练期间添加字符到词汇表
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

    def __getitem__(self, idx: int) -> tuple:
        sentence = self.sentences[idx]
        label = self.labels[idx]
        
        encoded_sentence = [self.vocab.char_to_idx.get(char, self.vocab.char_to_idx['<UNK>']) 
                          for char in sentence]
        encoded_label = [self.vocab.tag_to_idx[tag] for tag in label]
        
        return torch.tensor(encoded_sentence), torch.tensor(encoded_label)

def create_mask(seq: torch.Tensor) -> torch.Tensor:
    return (seq != 0).unsqueeze(-2)
#定义数据加载器的collate_fn函数
def collate_fn(batch: List[tuple]) -> tuple:
    sentences, labels = zip(*batch)
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    mask = create_mask(sentences)
    return sentences, labels, mask
#定义WarmupCosineSchedule类
class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.current_step / self.warmup_steps * self.optimizer.param_groups[0]['lr']
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = max(self.min_lr, 
                    0.5 * (1 + math.cos(math.pi * progress)) * self.optimizer.param_groups[0]['lr'])
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
#定义训练函数
def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, epochs: int = 5, device: str = 'cuda',
                grad_clip: float = 1.0, warmup_ratio: float = 0.1):
    model.to(device)
    model.train()
    
    # 计算总步数和warmup步数
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    
    # 创建学习率调度器
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps)
    
    # 用于记录最佳模型
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        print(f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (inputs, targets, mask) in enumerate(train_loader):
            # 将数据移到设备上
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs, mask)
            
            # 计算损失
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # 检查损失是否为nan
            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_idx+1}")
                continue
                
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # 优化器步进
            optimizer.step()
            
            # 学习率调度器步进
            curr_lr = scheduler.step()
            
            # 累积损失
            total_loss += loss.item()
            
            # 打印进度
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f'Batch {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Avg Loss: {avg_loss:.4f}, '
                      f'LR: {curr_lr:.6f}')
        
        # 计算epoch的平均损失
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f'Saving best model with loss {best_loss:.4f}')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, './cache/best_model.pt')
#定义分词函数
def segment_text(model: nn.Module, text: str, vocab: Vocabulary, 
                 device: str = 'cuda') -> List[str]:
    model.eval()
    
    chars = list(text)
    char_indices = [vocab.char_to_idx.get(char, vocab.char_to_idx['<UNK>']) 
                   for char in chars]
    
    tensor_input = torch.tensor(char_indices).unsqueeze(0).to(device)
    mask = create_mask(tensor_input).to(device)
    
    with torch.no_grad():
        outputs = model(tensor_input, mask)
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

def evaluate_model(model: nn.Module, 
                  test_loader: DataLoader, 
                  vocab: Vocabulary,
                  device: str = 'cuda') -> Dict[str, float]:
    """
    评估模型性能，计算准确率、召回率和F1分数
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        vocab: 词汇表对象
        device: 计算设备
        
    Returns:
        包含评估指标的字典
    """
    model.eval()
    total_correct = 0
    total_pred = 0
    total_gold = 0
    
    with torch.no_grad():
        for inputs, targets, mask in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            
            outputs = model(inputs, mask)
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

def evaluate_on_test_set(model: nn.Module,
                        test_path: str,
                        gold_path: str,
                        vocab: Vocabulary,
                        device: str = 'cuda',
                        batch_size: int = 32) -> Dict[str, float]:
    """
    在测试集上评估模型性能
    
    Args:
        model: 训练好的模型
        test_path: 测试数据路径
        gold_path: 黄金标准数据路径
        vocab: 词汇表对象
        device: 计算设备
        batch_size: 批次大小
        
    Returns:
        包含评估指标的字典
    """
    # 加载测试数据集
    test_dataset = ChineseSegmentationDataset(test_path, vocab, is_test=True)
    test_loader = DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           collate_fn=collate_fn)
    
    # 评估模型
    metrics = evaluate_model(model, test_loader, vocab, device)
    
    return metrics

def main():
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 初始化词表
    vocab = Vocabulary()
    
    # 加载训练数据
    train_path = '../dataset/icwb2-data/training/msr_training.utf8'
    train_dataset = ChineseSegmentationDataset(train_path, vocab)
    
    # 创建数据加载器，使用较小的batch_size
    train_loader = DataLoader(train_dataset, 
                            batch_size=16,  # 减小batch size
                            shuffle=True, 
                            collate_fn=collate_fn)
    
    # 初始化模型，减小模型规模
    model = ChineseSegmentationTransformer(
        vocab_size=vocab.get_vocab_size(),
        tag_size=vocab.get_tag_size(),
        d_model=128,  # 减小模型维度
        num_heads=4,  # 减少注意力头数
        num_layers=4,  # 减少层数
        d_ff=512,     # 减小前馈网络维度
        dropout=0.2   # 增加dropout
    )
    
    # 定义损失函数，添加标签平滑
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    # 使用较小的初始学习率
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01  # 添加权重衰减
    )
    
    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_loader, criterion, optimizer, epochs=10, device=device)

    test_path='../dataset/icwb2-data/testing/msr_test.utf8'
    gold_path='../dataset/icwb2-data/gold/msr_test_gold.utf8'
    # 在测试集上评估模型
    print("在测试集上评估模型...")
    metrics = evaluate_on_test_set(model, test_path, gold_path, vocab, device)
    print(f"测试集指标:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'model_config': {
            'vocab_size': vocab.get_vocab_size(),
            'tag_size': vocab.get_tag_size(),
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 6
        }
    }, './model/Transformers-based/chinese_segmentation_transformer.pt')

if __name__ == "__main__":
    main()