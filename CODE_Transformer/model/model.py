import os
import torch
import torch.nn as nn

#from metric import evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #这里保留的是参考代码的实现

class Trans(nn.module):
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
        #self.hidden_network=
    def getmask(self,i):
        return

    def encoder(self, input):
        output = input
        return output


    def decoder(self,src,tgt,mask):
        output = tgt
        return output


    def forward(self,melody,chord,mask):
        src = self.encoder(melody)
        lent = len(melody)
        output = []
        if self.mode == train :#
            tgt = chord
            for i in range(0,lent):
                mask = self.getmask(i)# to be finished
                output = self.decoder(src,tgt,mask)

        else:
            tgt = []
            for i in range(0,lent):
                mask = self.getmask(i)
                tgt = self.decoder(src,tgt,mask)
            output = tgt
        output_p = output# to be finished
        return output_p


def train(model_name):
    model_name.train()
    return


def evaluate(model_name):
    model_name.eval()
    return
