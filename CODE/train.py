import torch
import numpy as np
from model import LSTM
import torch.nn as nn
import torch.optim as optim
from metric_new import evaluation
from dataset import MidiDataset
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

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
            print(pred)
            avg=avg+evaluation(pred,chord) #调用验证函数
            cnt=cnt+1
        print('Verify set average accuracy:',avg/cnt)
    return model

if __name__ == '__main__':
    #尝试写的main函数，为了debug qwq，目前还在debug中
    data_path = 'output.pkl'
    data_set = MidiDataset(data_path)
    #-----转成tensor-----
    input_batch=[]
    target_batch=[]
    for melody,chord in data_set:
        input_batch.append(melody)
        target_batch.append(chord)

    input_batch,target_batch=torch.LongTensor(input_batch),torch.LongTensor(target_batch)
    #划分测试集
    x_train,x_test,y_train,y_test=train_test_split(input_batch, target_batch, test_size=0.2, random_state=0)
    x_train=torch.tensor(x_train).to(torch.float32)
    y_train=torch.tensor(y_train).to(torch.float32)
    x_test=torch.tensor(x_test).to(torch.float32)
    y_test=torch.tensor(y_test).to(torch.float32)
    #-----转化完成-----

    train_set = Data.TensorDataset(x_train,y_train)
    verify_set = Data.TensorDataset(x_test,y_test)
    train(train_set,verify_set)
