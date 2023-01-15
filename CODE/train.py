import torch
import numpy as np
from model import LSTM
import torch.nn as nn
import torch.optim as optim
from metric_new import evaluation
import dataset
#再import一个evaluate函数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #这里保留的是参考代码的实现

def train(train_loader,verify_loader): #train_loader是用来训练的数据集, verify是验证集
   #论文提供的参数：
    # 2层LSTM
    # 2 hidden layers
    # 128 BLSTM units
    # time distribute output layer with 24 units (output_size? 根据我们自己的调整应该为12)
    # dropout rate = 0.2
    # hidden_size = 128?
    # 是BLSTM所以应该 bidirectional = true
    model = LSTM(input_size=12, hidden_size=128, output_size=12, num_layers=2, bidirectional=True, dropout=0.2)

    #论文提供的描述
    # We use minibatch gradient descent with categorical cross entropy as the cost function and Adam as the optimize
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-3)

    MAX_EPOCH=1000 #随便设的一个值，根据数据集改
    for epoch in range(MAX_EPOCH):
        for melody, chord in train_loader:
            pred=model(melody)
            loss=criterion(pred,chord)

            if (epoch+1)%10==0:
                print('Epoch: ','%04d'%(epoch+1),'loss = ','{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #verify, 输出一个平均正确率？
        avg=0
        cnt=0
        for melody, chord in verify_loader:
            pred=model(melody)
            avg=avg+evaluation(pred,chord) #调用验证函数
            cnt=cnt+1
        print('Verify set average accuracy:',avg/cnt)
    return model
