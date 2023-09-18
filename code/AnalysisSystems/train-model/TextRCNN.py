# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'TextRCNN'
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

        self.dropout = 1.0  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 10  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 256  # lstm隐藏单元数
        self.num_layers = 1  # lstm层数

'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        if config.embedding_pretrained is not None:  # 加载初始化好的预训练词/字嵌入矩阵  微调funetuning
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:  # 否则随机初始化词/字嵌入矩阵 指定填充对应的索引
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        # 单层双向lstm batch_size为第一维度
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)

        self.maxpool = nn.MaxPool1d(config.pad_size)  # 沿长度方向作全局最大池化
        # 输出层
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        x, _ = x  # （batch,seq_len）
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)  # (batch_size,seq_len,hidden_size*2)
        out = torch.cat((embed, out), 2)  # 把词嵌入和lstm输出进行拼接 (batch,seq_len.embed+hidden_size*2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)  # (batch,embed+hidden_size*2,seq_len)
        out = self.maxpool(out).squeeze()  # （batch,embed+hidden_size*2）
        out = self.fc(out)  # (batch,classes)
        return out