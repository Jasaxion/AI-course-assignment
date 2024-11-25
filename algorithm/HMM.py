import os
import json
from collections import defaultdict
from typing import List, Tuple, Dict

class HMMTokenizer:
    def __init__(self):
        self.transition_prob = defaultdict(lambda: defaultdict(float))
        self.emission_prob = defaultdict(lambda: defaultdict(float))
        self.start_prob = defaultdict(float)
        self.states = ['B', 'M', 'E', 'S']
        self.state_count = defaultdict(int)
        self.char_count = defaultdict(int)

    def train(self, training_file: str) -> None:
        """
        训练HMM分词模型
        Args:
            training_file: 训练文件路径
        """
        with open(training_file, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                previous_state = None
                for word in words:
                    if len(word) == 1:
                        self._update_prob(previous_state, 'S', word)
                        previous_state = 'S'
                    else:
                        self._update_prob(previous_state, 'B', word[0])
                        previous_state = 'B'
                        for char in word[1:-1]:
                            self._update_prob(previous_state, 'M', char)
                            previous_state = 'M'
                        self._update_prob(previous_state, 'E', word[-1])
                        previous_state = 'E'

        # 归一化概率
        self._normalize_prob(self.start_prob)
        for state in self.states:
            self._normalize_prob(self.transition_prob[state])
            self._normalize_prob(self.emission_prob[state])

    def _update_prob(self, previous_state: str, current_state: str, char: str) -> None:
        """
        更新概率矩阵
        """
        if previous_state is None:
            self.start_prob[current_state] += 1
        else:
            self.transition_prob[previous_state][current_state] += 1
        self.emission_prob[current_state][char] += 1
        self.state_count[current_state] += 1
        self.char_count[char] += 1

    def _normalize_prob(self, prob_dict: Dict) -> None:
        """
        概率归一化
        """
        total = sum(prob_dict.values())
        if total > 0:
            for key in prob_dict:
                prob_dict[key] /= total

    def viterbi(self, sentence: str) -> List[str]:
        """
        维特比算法实现
        """
        V = [{}]
        path = {}

        # 初始化
        for state in self.states:
            V[0][state] = self.start_prob[state] * self.emission_prob[state].get(sentence[0], 1e-10)
            path[state] = [state]

        # 运行Viterbi算法
        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            for state in self.states:
                (prob, prev_state) = max(
                    (V[t-1][prev_state] * 
                     self.transition_prob[prev_state][state] * 
                     self.emission_prob[state].get(sentence[t], 1e-10), 
                     prev_state) 
                    for prev_state in self.states
                )
                V[t][state] = prob
                new_path[state] = path[prev_state] + [state]

            path = new_path

        # 获取最佳路径
        n = len(sentence) - 1
        (prob, state) = max((V[n][state], state) for state in self.states)
        return path[state]

    def tokenize(self, sentence: str) -> List[str]:
        """
        对句子进行分词
        """
        if not sentence:
            return []
            
        state_sequence = self.viterbi(sentence)
        tokens = []
        word = ""
        
        for char, state in zip(sentence, state_sequence):
            if state == 'B':
                word = char
            elif state == 'M':
                word += char
            elif state == 'E':
                word += char
                tokens.append(word)
                word = ""
            elif state == 'S':
                tokens.append(char)
                
        # 处理最后一个词
        if word:
            tokens.append(word)
            
        return tokens

    def save_model(self, model_dir: str) -> None:
        """
        保存模型到指定目录
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # 将defaultdict转换为普通dict以便序列化
        model_data = {
            'transition_prob': {k: dict(v) for k, v in self.transition_prob.items()},
            'emission_prob': {k: dict(v) for k, v in self.emission_prob.items()},
            'start_prob': dict(self.start_prob),
            'state_count': dict(self.state_count),
            'char_count': dict(self.char_count)
        }
        
        with open(os.path.join(model_dir, 'model.json'), 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

    def load_model(self, model_dir: str) -> None:
        """
        从指定目录加载模型
        """
        model_path = os.path.join(model_dir, 'model.json')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
            
        # 将普通dict转换回defaultdict
        self.transition_prob = defaultdict(lambda: defaultdict(float))
        self.emission_prob = defaultdict(lambda: defaultdict(float))
        self.start_prob = defaultdict(float)
        self.state_count = defaultdict(int)
        self.char_count = defaultdict(int)
        
        for k, v in model_data['transition_prob'].items():
            self.transition_prob[k].update(v)
        for k, v in model_data['emission_prob'].items():
            self.emission_prob[k].update(v)
        self.start_prob.update(model_data['start_prob'])
        self.state_count.update(model_data['state_count'])
        self.char_count.update(model_data['char_count'])

def evaluate(tokenizer: HMMTokenizer, test_file: str, gold_file: str) -> Tuple[float, float, float]:
    """
    评估分词器性能
    Args:
        tokenizer: HMM分词器实例
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
    tokenizer = HMMTokenizer()
    
    # 训练模型
    training_file = "../dataset/icwb2-data/training/msr_training.utf8"
    tokenizer.train(training_file)
    
    # 保存模型
    model_dir = "./model/HMM"
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