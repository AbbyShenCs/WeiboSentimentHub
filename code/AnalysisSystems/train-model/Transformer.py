import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'Transformer'
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

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 2000  # 若超过2000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度
        self.dim_model = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5  # 5头注意力机制
        self.num_encoder = 2  # 两个transformer encoder block
'''Attention Is All You Need'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        # 词/字嵌入
        if config.embedding_pretrained is not None:  # 加载初始化好的预训练词/字嵌入矩阵  微调funetuning
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:  # 否则随机初始化词/字嵌入矩阵 指定填充对应的索引
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        # 位置编码
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)

        # transformer encoder block
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        # 多个transformer encoder block
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(config.num_encoder)])

        # 输出层
        self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)
        # self.fc2 = nn.Linear(config.last_hidden, config.num_classes)
        # self.fc1 = nn.Linear(config.dim_model, config.num_classes)

    def forward(self, x):
        out = self.embedding(x[0])  # (batch,seq_len) -> (batch,seq_len,embed)
        out = self.postion_embedding(out)  # 添加位置编码 (batch,seq_len,embed)
        for encoder in self.encoders:  # 通过多个encoder block
            out = encoder(out)  # (batch,seq_len,dim_model)
        out = out.view(out.size(0), -1)  # （batch,seq_len*dim_model）
        # out = torch.mean(out, 1)
        out = self.fc1(out)  # (batch,classes)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        # 多头注意力机制
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        # 两个全连接层
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):  # x (batch,seq_len,embed_size)  embed_size = dim_model
        out = self.attention(x)  # 计算多头注意力结果 (batch,seq_len,dim_model)
        out = self.feed_forward(out)  # 通过两个全连接层增加 非线性转换能力 (batch,seq_len,dim_model)
        return out


class Positional_Encoding(nn.Module):


    # 位置编码
    def __init__(self, embed, pad_size, dropout, device):
        # 实例化nn中预定义的Dropout层, 并将dropout传入其中, 获得对象self.dropout
        self.dropout = nn.Dropout(dropout)
        # 考虑如何将这些位置信息加入到位置编码矩阵中，
        # 最简单思路就是先将pad_size x 1的绝对位置矩阵， 变换成pad_size x pad_size形状，然后覆盖原来的初始位置编码矩阵即可，
        # 要做这种矩阵变换，就需要一个1xembed形状的变换矩阵div_term，我们对这个变换矩阵的要求除了形状外，
        # 还希望它能够将自然数的绝对位置编码缩放成足够小的数字，有助于在之后的梯度下降过程中更快的收敛.  这样我们就可以开始初始化这个变换矩阵了.
        # 首先使用arange获得一个自然数矩阵， 但是细心的同学们会发现， 我们这里并没有按照预计的一样初始化一个1xpad_size的矩阵，
        # 而是有了一个跳跃，只初始化了一半即1xpad_size/2 的矩阵。 为什么是一半呢，其实这里并不是真正意义上的初始化了一半的矩阵，
        # 我们可以把它看作是初始化了两次，而每次初始化的变换矩阵会做不同的处理，第一次初始化的变换矩阵分布在正弦波上， 第二次初始化的变换矩阵分布在余弦波上，
        # 并把这两个矩阵分别填充在位置编码矩阵的偶数和奇数位置上，组成最终的位置编码矩阵.
        super(Positional_Encoding, self).__init__()
        self.device = device
        # 利用sin cos生成绝对位置编码
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])



    def forward(self, x):
        # token embedding + 绝对位置编码
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        # 再通过dropout
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''

    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        # Q与K的第2、3维转置计算内积  (batch*num_head,seq_len,seq_len)
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:  # 作缩放 减小结果的方差
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)  # 转换为权重
        context = torch.matmul(attention, V)  # 再与V运算 得到结果 (batch*num_head,seq_len,dim_head)
        return context


class Multi_Head_Attention(nn.Module):
    # 多头注意力机制 encoder block的第一部分
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head  # 头数
        assert dim_model % num_head == 0  # 必须整除
        self.dim_head = dim_model // self.num_head
        # 分别通过三个Dense层 生成Q、K、V
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        # Attention计算
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        # 层归一化
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):  # （batch,seq_len,dim_model）
        batch_size = x.size(0)
        # Q,K,V维度 (batch,seq_len,dim_head*num_head)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        # 沿第三个维度进行切分 切分为num_head份 再沿第一个维度拼接 多个注意力头并行计算
        # Q,K,V维度 (batch*num_head,seq_len,dim_head)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子 dim_head的开放取倒数 对内积结果进行缩放 减小结果的方差 有利于训练
        # attention计算 多个注意力头并行计算（矩阵运算）
        context = self.attention(Q, K, V, scale)

        # 多头注意力计算结果 沿第一个维度进行切分 再沿第三个维度拼接 转为原来的维度(batch,seq_len,dim_head*num_head)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)

        out = self.fc(context)  # (batch,seq_len,dim_model)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    # encoder block的第二部分
    def __init__(self, dim_model, hidden, dropout=0.0):
        # 定义两个全连接层 多头注意力的计算结果 通过两个全连接层 增加非线性
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):  # （batch,seq_len,dim_model）
        out = self.fc1(x)  # （batch,seq_len,hidden）
        out = F.relu(out)
        out = self.fc2(out)  # （batch,seq_len,dim_model）
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)  # 层归一化
        return out