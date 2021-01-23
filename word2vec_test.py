from gensim.models import word2vec
import jieba
# from src.pre_data import load_raw_data
from src.pre_data import *
import os
if not os.path.exists("data/processed.txt"):
    data = load_raw_data("data/Math_23K.json")
    print("Transfer numbers...")
    for question in data:
        pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
        seg = question['segmented_text'].split(' ')
        write_words = []
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                write_words.append("NUM")
                if pos.end() < len(s):
                    write_words.append(s[pos.end():])
            else:
                write_words.append(s)

        with open('data/processed.txt', 'a', encoding='utf-8') as ff:
            ff.write(' '.join(write_words))  # 词汇用空格分开
else:
    # 加载语料
    sentences = word2vec.Text8Corpus('data/processed.txt')
    print(sentences)
    # 训练模型
    model = word2vec.Word2Vec(sentences)

    # 选出最相似的10个词
    for e in model.most_similar(positive=['单价'], topn=10):
       print(e[0], e[1])






