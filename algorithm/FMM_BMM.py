#中文分词传统算法，前向最大匹配和后向最大匹配
import os

class FMM_BMM:
    def __init__(self, dictionary, max_len=5):
        self.dictionary = dictionary
        self.max_len = max_len

    # 前向最大匹配算法
    def forward_max_matching(self, text):
        result = []
        text_len = len(text)
        index = 0

        while index < text_len:
            matched = False
            for length in range(self.max_len, 0, -1):
                if index + length <= text_len:
                    word = text[index:index + length]
                    if word in self.dictionary:
                        result.append(word)
                        index += length
                        matched = True
                        break
            if not matched:
                result.append(text[index])
                index += 1

        return result

    # 后向最大匹配算法
    def backward_max_matching(self, text):
        result = []
        text_len = len(text)
        index = text_len

        while index > 0:
            matched = False
            for length in range(self.max_len, 0, -1):
                if index - length >= 0:
                    word = text[index - length:index]
                    if word in self.dictionary:
                        result.insert(0, word)
                        index -= length
                        matched = True
                        break
            if not matched:
                result.insert(0, text[index - 1])
                index -= 1

        return result