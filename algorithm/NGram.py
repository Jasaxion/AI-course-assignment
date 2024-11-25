import os
import json
from collections import defaultdict
from typing import List, Tuple, Dict
from math import log

class NgramTokenizer:
    def __init__(self, n=2):
        """
        初始化N-gram分词器
        Args:
            n: N-gram的N值，默认为2（bigram）
        """
        self.n = n
        self.word_freq = defaultdict(int)  # 词频统计
        self.word_count = 0  # 总词数
        self.char_freq = defaultdict(int)  # 字符频率
        self.ngram_freq = defaultdict(int)  # N-gram频率
        self.min_freq = 2  # 最小词频阈值
        self.max_word_length = 20  # 最大词长度

    def train(self, training_file: str) -> None:
        """
        训练N-gram分词模型
        Args:
            training_file: 训练文件路径
        """
        with open(training_file, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                for word in words:
                    # 更新词频
                    self.word_freq[word] += 1
                    self.word_count += 1
                    
                    # 更新字符频率
                    for char in word:
                        self.char_freq[char] += 1
                    
                    # 更新N-gram频率
                    if len(word) >= self.n:
                        for i in range(len(word) - self.n + 1):
                            ngram = word[i:i + self.n]
                            self.ngram_freq[ngram] += 1

    def _calculate_word_probability(self, word: str) -> float:
        """
        计算词的概率得分
        使用改进的计算方法，结合词频和N-gram概率
        """
        if not word:
            return float('-inf')
            
        # 如果是已知词，直接返回词频对数概率
        if word in self.word_freq:
            return log(self.word_freq[word] + 1) - log(self.word_count + len(self.word_freq))
            
        # 对未知词，使用N-gram概率
        score = 0
        if len(word) >= self.n:
            ngram_count = 0
            for i in range(len(word) - self.n + 1):
                ngram = word[i:i + self.n]
                if ngram in self.ngram_freq:
                    score += log(self.ngram_freq[ngram] + 1)
                    ngram_count += 1
                else:
                    score -= 1  # 惩罚未见过的N-gram
            
            if ngram_count > 0:
                score /= ngram_count
        else:
            # 对于短于N的词，使用字符频率
            for char in word:
                score += log(self.char_freq[char] + 1) - log(sum(self.char_freq.values()) + len(self.char_freq))
                
        return score

    def _segment_dp(self, sentence: str) -> List[str]:
        """
        使用动态规划算法进行分词
        """
        n = len(sentence)
        if n == 0:
            return []
            
        # dp[i]存储从开始到位置i的最大概率分词方案
        dp = [(float('-inf'), 0) for _ in range(n + 1)]
        dp[0] = (0, 0)
        
        # 动态规划求解最优分词
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_word_length), i):
                word = sentence[j:i]
                prob = self._calculate_word_probability(word)
                if dp[j][0] + prob > dp[i][0]:
                    dp[i] = (dp[j][0] + prob, j)
        
        # 回溯获取分词结果
        tokens = []
        i = n
        while i > 0:
            j = dp[i][1]
            tokens.insert(0, sentence[j:i])
            i = j
            
        return tokens

    def tokenize(self, sentence: str) -> List[str]:
        """
        对句子进行分词
        """
        return self._segment_dp(sentence)

    def save_model(self, model_dir: str) -> None:
        """
        保存模型到指定目录
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        model_data = {
            'n': self.n,
            'word_freq': dict(self.word_freq),
            'word_count': self.word_count,
            'char_freq': dict(self.char_freq),
            'ngram_freq': dict(self.ngram_freq),
            'min_freq': self.min_freq,
            'max_word_length': self.max_word_length
        }
        
        with open(os.path.join(model_dir, 'ngram_model.json'), 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

    def load_model(self, model_dir: str) -> None:
        """
        从指定目录加载模型
        """
        model_path = os.path.join(model_dir, 'ngram_model.json')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
            
        self.n = model_data['n']
        self.word_freq = defaultdict(int, model_data['word_freq'])
        self.word_count = model_data['word_count']
        self.char_freq = defaultdict(int, model_data['char_freq'])
        self.ngram_freq = defaultdict(int, model_data['ngram_freq'])
        self.min_freq = model_data['min_freq']
        self.max_word_length = model_data['max_word_length']

def evaluate(tokenizer: NgramTokenizer, test_file: str, gold_file: str) -> Tuple[float, float, float]:
    """
    评估分词器性能
    Args:
        tokenizer: N-gram分词器实例
        test_file: 测试文件路径
        gold_file: 金标准文件路径
    Returns:
        precision: 准确率
        recall: 召回率
        f1_score: F1分数
    """
    # 读取测试文件并分词
    predicted_words = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = tokenizer.tokenize(line)
                predicted_words.extend(tokens)

    # 读取金标准文件
    gold_words = []
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            gold_words.extend(words)

    # 计算评估指标
    correct = len(set(predicted_words) & set(gold_words))
    total_predicted = len(predicted_words)
    total_gold = len(gold_words)

    precision = correct / total_predicted if total_predicted > 0 else 0
    recall = correct / total_gold if total_gold > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def main():
    # 示例用法
    tokenizer = NgramTokenizer(n=2)  # 使用bigram模型
    
    # 训练模型
    training_file = "../dataset/icwb2-data/training/msr_training.utf8"
    tokenizer.train(training_file)
    
    # 保存模型
    model_dir = "./model/N-Gram/ngram_chinsesegment_model"
    tokenizer.save_model(model_dir)
    
    # 加载模型
    tokenizer.load_model(model_dir)
    
    # 评估模型
    test_file = "../dataset/icwb2-data/testing/msr_test.utf8"
    gold_file = "../dataset/icwb2-data/gold/msr_test_gold.utf8"
    precision, recall, f1_score = evaluate(tokenizer, test_file, gold_file)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

if __name__ == "__main__":
    main()