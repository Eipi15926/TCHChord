import torch
import torch.nn as nn
import torch.optim as optim
from model.metric import evaluation

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

    #def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional, dropout):
    def __init__(self, train_loader, verify_loader, test_loader, config):
        super(LSTM, self).__init__()
        #self.vocab_size = vocab_size #如果需要对齐的话才需要用嵌入层？目前来说我们应该不需要？
        #self.embedding_dim = embedding_dim
        self.eval_config=config['metric']
        self.save_place = config['save_dir']+config['save_name']

        config1=config['args']
        self.input_size = config1['input_size']
        self.hidden_size = config1['hidden_size']
        self.output_size = config1['output_size']
        self.num_layers = config1['num_layers'] #lstm层数
        self.bidirectional = config1['bidirectional'] #是否是双向的
        self.dropout = config1['dropout']

        config2=config['train']
        self.lr=config2['lr']
        self.max_epoch=config2['max_epochs']

        self.train_loader=train_loader
        self.verify_loader=verify_loader
        self.test_loader=test_loader
        
        #self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=word2idx['<PAD>'])
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers, bidirectional=self.bidirectional)

        #论文参数：hidden layer是两层, 激活函数是tanh
        if self.bidirectional: #因为不确定BLSTM和LSTM实现上的区别，所以这里保留了参考的代码的写法
            self.hidden_network = nn.Sequential(
                nn.Dropout(self.dropout), 
                nn.Linear(self.hidden_size*2, self.hidden_size),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size,self.output_size), #两层hidden size不知道怎么设，先设了一个每次/2
                nn.Tanh())
        else:
            self.hidden_network = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size/2),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size/2,self.output_size),
                nn.Tanh())
        
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

    def Train(self): 
        #参数：train_loader是用来训练的数据集, verify是验证集, lr, Epoch数量
        #论文提供的描述
        # We use minibatch gradient descent with categorical cross entropy as the cost function and Adam as the optimize
        train_loader=self.train_loader
        verify_loader=self.verify_loader
        lr=self.lr
        MAX_EPOCH=self.max_epoch

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(self.parameters(),lr)

        for epoch in range(MAX_EPOCH):
            self.train()
            for melody, chord in train_loader:
                melody=torch.tensor(melody).to(torch.float32)
                chord=torch.tensor(chord).to(torch.float32)
                melody=melody.to(device)
                chord=chord.to(device)

                pred=self(melody)
                loss=criterion(pred,chord)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            print('Epoch: ','%04d'%(epoch+1),'loss = ','{:.6f}'.format(loss))
            #verify, 输出一个平均正确率
            avg=0
            cnt=0
            self.eval()
            for melody, chord in verify_loader:
                melody=torch.tensor(melody).to(torch.float32)
                chord=torch.tensor(chord).to(torch.float32)
                melody = melody.to(device)
                chord = chord.to(device)

                pred=self(melody)
                avg=avg+evaluation(pred,chord,self.eval_config) #调用验证函数
                cnt=cnt+1
            print('Verify set average accuracy:',avg/cnt)
            print('')
        torch.save(self,self.save_place)
        return


def gen_model(train_loader, verify_loader, test_loader, config):
    '''
    args: train_loader, verify_loader, test_loader, config
    '''
    model=LSTM(train_loader, verify_loader, test_loader, config)
    return model.to(device)