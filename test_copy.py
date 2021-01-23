#! -*- coding: utf-8 -*-
# 用Seq2Seq做小学数学应用题
# 数据集为ape210k：https://github.com/Chenny0808/ape210k
# Base版准确率为70%+，Large版准确率为73%+
# 实测环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.8.8
# 介绍链接：https://kexue.fm/archives/7809

from __future__ import division
import json, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from bert4keras.backend import keras, K, batch_gather
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from sympy import Integer

# 基本参数
maxlen = 192
batch_size = 32
epochs = 100

# bert配置
# config_path = '/root/kg/bert/uer/mixed_corpus_bert_base_model/bert_config.json'
# checkpoint_path = '/root/kg/bert/uer/mixed_corpus_bert_base_model/bert_model.ckpt'
# dict_path = '/root/kg/bert/uer/mixed_corpus_bert_base_model/vocab.txt'
# bert配置
config_path = '/content/drive/MyDrive/BERT/bert_config.json'
checkpoint_path = '/content/drive/MyDrive/BERT/bert_model.ckpt'
dict_path = '/content/drive/MyDrive/BERT/vocab.txt'

def is_equal(a, b):
    """比较两个结果是否相等
    """
    a = round(float(a), 6)
    b = round(float(b), 6)
    return a == b


def remove_bucket(equation):
    """去掉冗余的括号
    """
    l_buckets, buckets = [], []
    for i, c in enumerate(equation):
        if c == '(':
            l_buckets.append(i)
        elif c == ')':
            buckets.append((l_buckets.pop(), i))
    eval_equation = eval(equation)
    for l, r in buckets:
        new_equation = '%s %s %s' % (
            equation[:l], equation[l + 1:r], equation[r + 1:]
        )
        try:
            if is_equal(eval(new_equation.replace(' ', '')), eval_equation):
                equation = new_equation
        except:
            pass
    return equation.replace(' ', '')


def load_data(filename):
    """读取训练数据，并做一些标准化，保证equation是可以eval的
    参考：https://kexue.fm/archives/7809
    """
    D = []
    for l in open(filename):
        l = json.loads(l)
        question, equation, answer = l['original_text'], l['equation'], l['ans']
        # 处理带分数
        question = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', question)
        equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', equation)
        answer = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', answer)
        equation = re.sub('(\d+)\(', '\\1+(', equation)
        answer = re.sub('(\d+)\(', '\\1+(', answer)
        # 分数去括号
        question = re.sub('\((\d+/\d+)\)', '\\1', question)
        # 处理百分数
        equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
        answer = re.sub('([\.\d]+)%', '(\\1/100)', answer)
        # 冒号转除号、剩余百分号处理
        equation = equation.replace(':', '/').replace('%', '/100')
        answer = answer.replace(':', '/').replace('%', '/100')
        if equation[:2] == 'x=':
            equation = equation[2:]
        try:
            if is_equal(eval(equation), eval(answer)):
                D.append((question, remove_bucket(equation), answer))
        except:
            continue
    return D


# 加载数据集
train_data = load_data('/content/drive/MyDrive/dataset/train23k.json')
valid_data = load_data('/content/drive/MyDrive/dataset/valid23k.json')
# test_data = load_data('/content/drive/MyDrive/dataset/test.ape.json')

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

# def generate_copy_labels(source, target):
#     """构建copy机制对应的label
#     """
#     mapping = longest_common_subsequence(source, target)[1]
#     source_labels = [0] * len(source)
#     target_labels = [0] * len(target)
#     i0, j0 = -2, -2
#     for i, j in mapping:
#         if i == i0 + 1 and j == j0 + 1:
#             source_labels[i] = 2
#             target_labels[j] = 2
#         else:
#             source_labels[i] = 1
#             target_labels[j] = 1
#         i0, j0 = i, j
#     return source_labels, target_labels

def random_masking(token_ids):
    """对输入进行随机mask，增加泛化能力
    """
    rands = np.random.random(len(token_ids))
    return [
        t if r > 0.15 else np.random.choice(token_ids)
        for r, t in zip(rands, token_ids)
    ]

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_output_ids, batch_labels = [], []
        for is_end, (question, equation, answer) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                question, equation, maxlen=maxlen
            )
            idx = token_ids.index(tokenizer._token_end_id) + 1

            masked_token_ids = random_masking(token_ids)

            source_labels, target_labels = masked_token_ids[:idx], token_ids[idx:]
            labels = source_labels + target_labels[1:]
            batch_token_ids.append(masked_token_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(token_ids)
            batch_labels.append(labels)


            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [
                    batch_token_ids, batch_segment_ids\
                    batch_output_ids, batch_labels
                ], None
                batch_token_ids, batch_segment_ids = [], []
                batch_output_ids, batch_labels = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    # def compute_loss(self, inputs, mask=None):
    #     y_true, y_mask, y_pred = inputs
    #     y_true = y_true[:, 1:]  # 目标token_ids
    #     y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
    #     y_pred = y_pred[:, :-1]  # 预测序列，错开一位
    #     loss = K.sparse_categorical_crossentropy(y_true, y_pred)
    #     loss = K.sum(loss * y_mask) / K.sum(y_mask)
    #     return loss
    def compute_loss(self, inputs, mask=None):
        seq2seq_loss = self.compute_seq2seq_loss(inputs, mask)
        copy_loss = self.compute_copy_loss(inputs, mask)
        self.add_metric(seq2seq_loss, 'seq2seq_loss')
        self.add_metric(copy_loss, 'copy_loss')
        return seq2seq_loss + 2 * copy_loss

    def compute_seq2seq_loss(self, inputs, mask=None):
        y_true, y_mask, _, y_pred, _ = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, :-1] * y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        # 正loss
        pos_loss = batch_gather(y_pred, y_true[..., None])[..., 0]
        # 负loss
        y_pred = tf.nn.top_k(y_pred, k=k_sparse)[0]
        neg_loss = K.logsumexp(y_pred, axis=-1)
        # 总loss
        loss = neg_loss - pos_loss
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def compute_copy_loss(self, inputs, mask=None):
        _, y_mask, y_true, _, y_pred = inputs
        y_mask = K.cumsum(y_mask[:, ::-1], axis=1)[:, ::-1]
        y_mask = K.cast(K.greater(y_mask, 0.5), K.floatx())
        y_mask = y_mask[:, 1:]  # mask标记，减少一位
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(2e-5))
model.summary()


class AutoSolve(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(model).predict([token_ids, segment_ids])

    def generate(self, text, topk=1):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids).replace(' ', '')


autosolve = AutoSolve(start_id=None, end_id=tokenizer._token_end_id, maxlen=64)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)  # 评测模型
        if metrics['acc'] >= self.best_acc:
            self.best_acc = metrics['acc']
            model.save_weights('./best_model.weights')  # 保存模型
        metrics['best_acc'] = self.best_acc
        print('valid_data:', metrics)

    def evaluate(self, data, topk=1):
        total, right = 0.0, 0.0
        for question, equation, answer in tqdm(data):
            total += 1
            pred_equation = autosolve.generate(question, topk)
            try:
                right += int(is_equal(eval(pred_equation), eval(answer)))
            except:
                pass
        return {'acc': right / total}


def predict(in_file, out_file, topk=1):
    """输出预测结果到文件
    该函数主要为比赛 https://www.datafountain.cn/competitions/467 所写，
    主要是读取该比赛的测试集，然后预测equation，并且根据不同的问题输出不同格式的答案，
    out_file可以直接提交到线上评测，线上准确率可以达到38%+。
    """
    fw = open(out_file, 'w', encoding='utf-8')
    raw_data = pd.read_csv(in_file, header=None, encoding='utf-8')
    for i, question in tqdm(raw_data.values):
        question = re.sub('(\d+)_(\d+/\d+)', '(\\1+\\2)', question)
        pred_equation = autosolve.generate(question, topk)
        if '.' not in pred_equation:
            pred_equation = re.sub('([\d]+)', 'Integer(\\1)', pred_equation)
        try:
            pred_answer = eval(pred_equation)
        except:
            pred_answer = np.random.choice(21) + 1
        if '.' in pred_equation:
            if u'百分之几' in question:
                pred_answer = pred_answer * 100
            pred_answer = round(pred_answer, 2)
            if int(pred_answer) == pred_answer:
                pred_answer = int(pred_answer)
            if (
                re.findall(u'多少[辆|人|个|只|箱|包本|束|头|盒|张]', question) or
                re.findall(u'几[辆|人|个|只|箱|包|本|束|头|盒|张]', question)
            ):
                if re.findall(u'至少|最少', question):
                    pred_answer = np.ceil(pred_answer)
                elif re.findall(u'至多|最多', question):
                    pred_answer = np.floor(pred_answer)
                else:
                    pred_answer = np.ceil(pred_answer)
                pred_answer = int(pred_answer)
            pred_answer = str(pred_answer)
            if u'百分之几' in question:
                pred_answer = pred_answer + '%'
        else:
            pred_answer = str(pred_answer)
            if '/' in pred_answer:
                if re.findall('\d+/\d+', question):
                    a, b = pred_answer.split('/')
                    a, b = int(a), int(b)
                    if a > b:
                        pred_answer = '%s_%s/%s' % (a // b, a % b, b)
                else:
                    if re.findall(u'至少|最少', question):
                        pred_answer = np.ceil(eval(pred_answer))
                    elif re.findall(u'至多|最多', question):
                        pred_answer = np.floor(eval(pred_answer))
                    else:
                        pred_answer = np.ceil(eval(pred_answer))
                    pred_answer = str(int(pred_answer))
        fw.write(str(i) + ',' + pred_answer + '\n')
        fw.flush()
    fw.close()


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model.weights')