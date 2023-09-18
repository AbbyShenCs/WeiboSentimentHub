# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """FastText配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'FastText'
        # 训练集、验证集、测试集路径
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        # 数据集的所有类别
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]
        # 构建好的词/字典路径
        self.vocab_path = dataset + '/data/vocab.pkl'
        # 训练好的模型参数保存路径
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        # 模型日志保存路径
        self.log_path = dataset + '/log/' + self.model_name
        # 如果词/字嵌入矩阵不随机初始化 则加载初始化好的词/字嵌入矩阵 类别为float32 并转换为tensor 否则为None
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5  # 随机失活 丢弃率
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度
        self.hidden_size = 256  # 隐藏层大小
        self.n_gram_vocab = 250499  # n-gram词表大小
'''Bag of Tricks for Efficient Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        if config.embedding_pretrained is not None:  # 加载初始化好的预训练词/字嵌入矩阵  微调funetuning
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:  # 否则随机初始化词/字嵌入矩阵 指定填充对应的索引
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        # 分别随机初始化 bi-gram tri-gram对应的词嵌入矩阵
        self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.embed)
        # dropout
        self.dropout = nn.Dropout(config.dropout)
        # 隐层
        self.fc1 = nn.Linear(config.embed * 3, config.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        # 输出层
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x (uni-gram,seq_len,bi-gram,tri-gram)
        # 基于uni-gram、bi-gram、tri-gram对应的索引 在各自的词嵌入矩阵中查询 得到词嵌入
        # （batch,seq_len,embed）
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        # 三种嵌入进行拼接 (batch,seq,embed*3)
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        # 沿长度维 作平均 (batch,embed*3)
        out = out.mean(dim=1)
        # 通过fropout
        out = self.dropout(out)
        # 通过隐层 (batch,hidden_size)
        out = self.fc1(out)
        out = F.relu(out)
        # 输出层 (batch,classes)
        out = self.fc2(out)
        return out
