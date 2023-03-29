import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from model.metric import evaluation
from model.metric import evaluation_simple
from model.MyLossFunc import MyLossFunc

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") #这里保留的是参考代码的实现

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
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
        super(TransformerModel, self).__init__()
        #self.vocab_size = vocab_size #如果需要对齐的话才需要用嵌入层？目前来说我们应该不需要？
        #self.embedding_dim = embedding_dim
        self.eval_config=config['metric']
        self.save_place = config['save_dir']+config['save_name']
        if os.path.exists(config['save_dir']):
            pass
        else:
            os.makedirs(config['save_dir'])

        config1=config['args']
        self.config1 = config1
        self.input_size = config1['input_size']
        self.hidden_size = config1['hidden_size']
        self.output_size = config1['output_size']
        self.num_layers = config1['num_layers'] #lstm层数
        self.bidirectional = config1['bidirectional'] #是否是双向的
        self.dropout = config1['dropout']
        
        # transformer parameters
        self.d_model = config1['d_model']
        self.nhead = config1['nhead']
        self.d_hid = config1['d_hid']
        self.nlayers = config1['nlayers']

        config2=config['train']
        self.config2 = config2
        self.lr=config2['lr']
        self.max_epoch=config2['max_epochs']
        self.early_stop = config2['early_stop']
        self.loss_fn = config2['loss_fn']

        self.train_loader=train_loader
        self.verify_loader=verify_loader
        self.test_loader=test_loader
        self.batch_size = train_loader.batch_size
        
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        encoder_layers = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.d_hid, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.nlayers)
        # self.decoder = nn.Linear(self.d_model, self.output_size)
        self.decoder = nn.Sequential(nn.Linear(self.d_model, self.hidden_size),                                     
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout),
                                     nn.Linear(self.hidden_size, self.hidden_size//2),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout),
                                     nn.Linear(self.hidden_size//2, self.output_size))
        # self.lstm = nn.LSTM(input_size=self.input_size, 
        #                     hidden_size=self.hidden_size, 
        #                     batch_first=True, 
        #                     num_layers=self.num_layers, 
        #                     bidirectional=self.bidirectional)

        #论文参数：hidden layer是两层, 激活函数是tanh
        # if self.bidirectional: #因为不确定BLSTM和LSTM实现上的区别，所以这里保留了参考的代码的写法
        #     self.hidden_network = nn.Sequential(
        #         nn.Dropout(self.dropout), 
        #         nn.Linear(self.hidden_size*2, self.hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(self.dropout),
        #         nn.Linear(self.hidden_size, self.hidden_size//2),
        #         nn.ReLU(),
        #         nn.Dropout(self.dropout),
        #         nn.Linear(self.hidden_size//2, self.hidden_size//4),
        #         nn.ReLU(),
        #         nn.Dropout(self.dropout),
        #         nn.Linear(self.hidden_size//4,self.output_size)) #两层hidden size不知道怎么设，先设了一个每次/2
                
        # else:
        #     self.hidden_network = nn.Sequential(
        #         nn.Dropout(self.dropout),
        #         nn.Linear(self.hidden_size, self.hidden_size/2),
        #         nn.Tanh(),
        #         nn.Dropout(self.dropout),
        #         nn.Linear(self.hidden_size/2,self.output_size),
        #         nn.Tanh())
        
    def forward(self, data, src_mask=None):
        #没有分成批所以应该是单个序列跑的没有batch
        #batch_size = data.size(0) #序列长度
        #print(batch_size)
        #print(data)
        #初始化一个h0,也即c0，在RNN中一个Cell输出的ht和Ct是相同的，而LSTM的一个cell输出的ht和Ct是不同的
        #维度[layers, hidden_len]
        # if self.bidirectional: #因为不太确定BLSTM和LSTM的具体实现区别所以这块是直接保留了参考的代码的内容
        #     h0 = torch.randn(self.num_layers*2, self.batch_size, self.hidden_size).to(device)
        #     c0 = torch.randn(self.num_layers*2, self.batch_size, self.hidden_size).to(device)
        # else:
        #     h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)
        #     c0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)
        # out,(_,_)= self.lstm(data, (h0,c0))
        out = self.pos_encoder(data)
        out = self.transformer_encoder(out, src_mask)
        output = self.decoder(out)
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

        if self.loss_fn == 'BCEWithLogitsLoss':
            pos_weight = self.config2['pos_weight']
            pos_weight = torch.tensor([pos_weight] * 12)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
        elif self.loss_fn == 'MyLossFunc':
            criterion = MyLossFunc().to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(self.parameters(),lr)

        stop_cnt=0
        pre_acc=0
        for epoch in range(MAX_EPOCH):
            self.train()
            self.batch_size = self.train_loader.batch_size
            for melody, chord in train_loader:
                # melody=torch.tensor(melody).to(torch.float32)
                # chord=torch.tensor(chord).to(torch.float32)
                # melody = torch.tensor(melody)
                # chord = torch.tensor(chord)
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
            self.batch_size = self.verify_loader.batch_size
            self.eval()
            for melody, chord in verify_loader:
                # melody=torch.tensor(melody).to(torch.float32)
                # chord=torch.tensor(chord).to(torch.float32)
                melody = melody.to(device)
                chord = chord.to(device)

                pred=self(melody)
                avg=avg+evaluation_simple(pred,chord,self.eval_config) #调用验证函数
                cnt=cnt+1
                
                # if cnt % 100 == 0:
                #     print(f'count: {cnt}')
                
            print('Verify set average accuracy:',avg/cnt)
            print('')
            if (epoch>1)&(pre_acc>avg/cnt):
                stop_cnt=stop_cnt+1
                if stop_cnt>=self.early_stop:
                    break
            else:
                stop_cnt=0
            pre_acc=avg/cnt
        torch.save(self,self.save_place)
        return

    def test(self):
        avg = 0
        cnt = 0
        self.batch_size = self.test_loader.batch_size
        self.eval()
        for melody, chord in self.test_loader:
            # melody=torch.tensor(melody).to(torch.float32)
            # chord=torch.tensor(chord).to(torch.float32)
            melody = melody.to(device)
            chord = chord.to(device)

            pred=self(melody)
            avg=avg+evaluation(pred,chord,self.eval_config) #调用验证函数
            cnt=cnt+1

            # if cnt % 100 == 0:
            #     print(f'count: {cnt}')
        print('Test set average accuracy:',avg/cnt)
        return

def gen_model(train_loader, verify_loader, test_loader, config):
    '''
    args: train_loader, verify_loader, test_loader, config
    '''
    model=TransformerModel(train_loader, verify_loader, test_loader, config)
    # model = torch.load('save/mymodel.pt')
    # model.max_epoch = 1000
    print(model)
    return model.to(device)
