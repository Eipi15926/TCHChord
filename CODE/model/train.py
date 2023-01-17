import torch
import numpy as np
from model import LSTM
from dataset import MidiDataset
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    #尝试写的main函数，为了debug qwq，目前还在debug中
    data_path = 'output.pkl'
    data_set = MidiDataset(data_path)
    #-----转成tensor-----
    input_batch=[]
    target_batch=[]
    '''
    mx=0
    temp=[]
    for i in range(0,12):
        temp.append(0)
    for melody,chord in data_set:
        while (len(melody)!=mx):
            melody.append(temp)
            chord.append(temp)
        input_batch.append(melody)
        target_batch.append(chord)

    input_batch,target_batch=torch.Tensor(input_batch),torch.Tensor(target_batch)
    #划分测试集
    x_train,x_test,y_train,y_test=train_test_split(input_batch, target_batch, test_size=0.2, random_state=0)
    '''

    #仅拿第一个数据跑来看能不能跑通
    x_train=data_set[0]
    y_train=data_set[1]
    x_test=x_train
    y_test=y_train

    x_train=torch.tensor(x_train).to(torch.float32)
    y_train=torch.tensor(y_train).to(torch.float32)
    x_test=torch.tensor(x_test).to(torch.float32)
    y_test=torch.tensor(y_test).to(torch.float32)
    #-----转化完成-----

    train_set = Data.TensorDataset(x_train,y_train)
    verify_set = Data.TensorDataset(x_test,y_test)

    model = LSTM(input_size=12, hidden_size=128, output_size=12, num_layers=2, bidirectional=True, dropout=0.2)
    model.train(train_set,verify_set,1e-3,10)
