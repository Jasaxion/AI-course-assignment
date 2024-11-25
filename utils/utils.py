# 加载词典
def load_dictionary(file_path):
    dictionary = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                dictionary.add(word)
    return dictionary