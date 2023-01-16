import torch
#import numpy as np
import torch.nn as nn
#from dataset import MidiDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #这里保留的是参考代码的实现
class LSTM(nn.Module):
    #论文提供的参数：
    # 2层LSTM
    # 2 hidden layers
    # 128 BLSTM units
    # time distribute output layer with 24 units (output_size? 根据我们自己的调整应该为12)
    # dropout rate = 0.2
    # hidden_size = 128?
    # 是BLSTM所以应该 bidirectional = true
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional, dropout):
        super(LSTM, self).__init__()
        #self.vocab_size = vocab_size #如果需要对齐的话才需要用嵌入层？目前来说我们应该不需要？
        #self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers #lstm层数
        self.bidirectional = bidirectional #是否是双向的
        self.dropout=dropout
        
        #self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=word2idx['<PAD>'])
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers, bidirectional=self.bidirectional)

        #论文参数：hidden layer是两层, 激活函数是tanh
        if self.bidirectional: #因为不确定BLSTM和LSTM实现上的区别，所以这里保留了参考的代码的写法
            self.hidden_network = nn.Sequential(
                nn.Dropout(self.dropout), 
                nn.Linear(self.hidden_size*2, self.hidden_size),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size,output_size), #两层hidden size不知道怎么设，先设了一个每次/2
                nn.Tanh())
            #self.fc = nn.Linear(hidden_size*2, num_classes) #这句是参考代码的实现
        else:
            self.hidden_network = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size/2),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size/2,output_size),
                nn.Tanh())
            #self.fc = nn.Linear(hidden_size, num_classes) #这句是参考代码的实现
        
    def forward(self, data):
        #没有分成批所以应该是单个序列跑的没有batch
        #batch_size = data.size(0) #序列长度
        #print(batch_size)
        #print(data)
        #初始化一个h0,也即c0，在RNN中一个Cell输出的ht和Ct是相同的，而LSTM的一个cell输出的ht和Ct是不同的
        #维度[layers, hidden_len]
        if self.bidirectional: #因为不太确定BLSTM和LSTM的具体实现区别所以这块是直接保留了参考的代码的内容
            h0 = torch.randn(self.num_layers*2, self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers*2, self.hidden_size).to(device)
        else:
            h0 = torch.randn(self.num_layers, self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers, self.hidden_size).to(device)
        out,(_,_)= self.lstm(data, (h0,c0))
        output = self.hidden_network(out) 
        chord_out = torch.sigmoid(output) #最后用sigmoid输出概率
        return chord_out
        