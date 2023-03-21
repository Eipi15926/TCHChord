import os
import torch
import torch.nn as nn
import torch.optim as optim
#from metric import evaluation
from metric import evaluation_simple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #这里保留的是参考代码的实现
class Trans(nn.Module):
    def __init__(self, train_loader, verify_loader, test_loader,config):
        super(Trans, self).__init__()
        self.eval_config=config['metric']
        self.save_place=config['save_dir']+config['save_name']
        if os.path.exists(self.save_place):
            pass
        else:
            os.makedirs(config['save_dir'])

        #arguments of model, should be modified according to need
        config1=config['args']
        self.nhead=config1['nhead']
        self.num_encoder_layers=config1['num_encoder_layers']
        self.num_decoder_layers=config1['num_decoder_layers']
        self.dim_feedforward=config1['dim_feedforward']
        self.dropout=config1['dropout']

        #arguments of training process, should be modified according to need
        config2=config['train']
        self.lr=config2['lr']
        self.max_epoch=config2['max_epochs']
        self.early_stop=config2['early_stop']

        self.train_loader=train_loader
        self.verify_loader=verify_loader
        self.test_loader=test_loader
        self.batch_size = train_loader.batch_size

        #define model
        #need some other arguments?
        self.trans=nn.Transformer(nhead=self.nhead,
                                  num_encoder_layers=self.num_encoder_layers,
                                  num_decoder_layers=self.num_decoder_layers,
                                  dim_feedforward=self.dim_feedforward,
                                  dropout=self.dropout)
        #self.hidden_network=

 
    def forward(self):
        return

    # 一个用于修改target的函数，在训练集里target就是chord
    def get_mask(self,mask):
        return mask


    def func(self,output_p):
        output = output_p
        return output


    def train(self, device):
        train_loader = self.train_loader
        verify_loader = self.verify_loader
        lr = self.lr
        MAX_EPOCH = self.max_epoch

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(self.parameters(), lr)

        for epoch in range(MAX_EPOCH):
            #trainning mode on
            self.train()
            batch_size = self.batch_size
            for melody, chord in train_loader:
                lent = len(chord)
                melody = melody.to(device)
                chord = chord.to(device)
                target = chord  # 对训练来说是这样的
                for i in range(lent):
                    mask = self.get_mask(i) #give i as the argument or what?
                    output_p = self(melody, target, mask)
                    output = self.func(output_p)

                loss = criterion(output, chord)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Epoch: ','%04d'%(epoch+1),'loss = ','{:.6f}'.format(loss))
            
            #verify, 输出一个平均正确率
            avg=0
            cnt=0
            self.batch_size = self.verify_loader.batch_size

            #evalation mode on
            self.eval()
            for melody, chord in verify_loader:
                melody = melody.to(device)
                chord = chord.to(device)
                target = [] #should be empty at first and add the latest prediction result each time?
                lent = len(chord)
                for i in range(lent):
                    mask = self.get_mask(i) #give i as the argument or what?
                    pred = self(melody,target,mask)
                    target = [] #update target
                avg = avg + evaluation_simple(pred,chord,self.eval_config)
                cnt = cnt+1

                # if cnt % 100 == 0:
                #     print(f'count: {cnt}')
                
            print('Verify set average accuracy:',avg/cnt)
            print('')
            if (epoch > 1) & (pre_acc > avg/cnt):
                stop_cnt = stop_cnt + 1
                if stop_cnt >= self.early_stop:
                    break
            else:
                stop_cnt=0
            pre_acc = avg/cnt

        #save model
        torch.save(self,self.save_place)
        return


    def test(self, device):
        avg=0
        cnt=0
        self.batch_size = self.verify_loader.batch_size

        #evalation mode on
        self.eval()
        for melody, chord in verify_loader:
            melody = melody.to(device)
            chord = chord.to(device)
            #should be random at first and add the latest prediction result each time?
            target = torch.rand(()) #remember to add size of the tensor
            lent = len(chord)
            for i in range(lent):
                mask = self.get_mask(i) #give i as the argument or what?
                pred = self(melody,target,mask)
                target = [] #update target
            avg = avg + evaluation_simple(pred,chord,self.eval_config)
            cnt = cnt+1

        print('Test set average accuracy:',avg/cnt)
        return

        '''
        train_loader = self.train_loader
        verify_loader = self.verify_loader
        lr = self.lr
        MAX_EPOCH = self.max_epoch

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(self.parameters(), lr)

        for epoch in range(MAX_EPOCH):
            self.batch_size = self.train_loader.batch_size
            for melody, chord, target, mask in train_loader:
                lent = len(chord)
                output = []  # 应该是tensor 注意要改一下
                melody = melody.to(device)
                chord = chord.to(device)
                target = chord  # 对训练来说是这样的
                for i in range(lent):
                    mask = get_mask(mask)
                    #坏事了这个怎么改
                    output_p = trmodel(melody, target, mask)
                    output = func(output_p)
                    target = output

                loss = criterion(output, chord)
                # 评估函数之后再改
                print("loss=", loss)
        '''
def gen_model(train_loader, verify_loader, test_loader, config):
    '''
    args: train_loader, verify_loader, test_loader, config
    '''
    #if need to load saved model
    #model = torch.load('save/mymodel.pt')
    
    #if generate new model
    model=Trans(train_loader, verify_loader, test_loader, config)
    model.max_epoch = 1000

    return model.to(device)