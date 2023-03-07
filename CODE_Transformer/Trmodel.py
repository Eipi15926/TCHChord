import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from metric import evaluation
from metric import evaluation_simple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 这里保留的是参考代码的实现

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class transformer(nn.Module):
    # d_model: Encoder和Decoder输入参数的特征维度。也就是词向量的维度。默认为512
    # nhead: 多头注意力机制中，head的数量。关于Attention机制，可以参考这篇文章。注意该值并不影响网络的深度和参数数量。默认值为8。
    # encoder_layers: TransformerEncoderLayer的数量。该值越大，网络越深，网络参数量越多，计算量越大。默认值为6
    # decoder_layers：TransformerDecoderLayer的数量。该值越大，网络越深，网络参数量越多，计算量越大。默认值为6
    # dim_feedforward：Feed Forward层（Attention后面的全连接网络）的隐藏层的神经元数量。该值越大，网络参数量越多，计算量越大。默认值为2048
    # dropout：dropout值。默认值为0.1
    # activation： Feed Forward层的激活函数。取值可以是string(“relu” or “gelu”)或者一个一元可调用的函数。默认值是relu
    # custom_encoder：自定义Encoder。若你不想用官方实现的TransformerEncoder，你可以自己实现一个。默认值为None
    # custom_decoder: 自定义Decoder。若你不想用官方实现的TransformerDecoder，你可以自己实现一个。
    # layer_norm_eps: Add&Norm层中，BatchNorm的eps参数值。默认为1e-5
    # batch_first：batch维度是否是第一个。如果为True，则输入的shape应为(batch_size, 词数，词向量维度)，否则应为(词数, batch_size, 词向量维度)。默认为False。这个要特别注意，因为大部分人的习惯都是将batch_size放在最前面，而这个参数的默认值又是False，所以会报错。
    # norm_first – 是否要先执行norm。

    # def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional, dropout):
    def __init__(self, train_loader, verify_loader, test_loader, config):
        super(transformer, self).__init__()
        # self.vocab_size = vocab_size #如果需要对齐的话才需要用嵌入层？目前来说我们应该不需要？
        # self.embedding_dim = embedding_dim
        self.eval_config = config['metric']
        self.save_place = config['save_dir'] + config['save_name']
        if os.path.exists(self.save_place):
            pass
        else:
            os.makedirs(config['save_dir'])

        config1 = config['args']
        self.d_model = config1['d_model']#512
        self.encoder_layers = config1['encoder_layers']#6
        self.decoder_layers = config1['decoder_layers']#6

        self.hidden_size = config1['hidden_size']
        self.output_size = config1['output_size']
        self.dropout = config1['dropout']

        config2 = config['train']
        self.lr = config2['lr']
        self.max_epoch = config2['max_epochs']
        self.early_stop = config2['early_stop']

        self.train_loader = train_loader
        self.verify_loader = verify_loader
        self.test_loader = test_loader
        self.batch_size = train_loader.batch_size

        # self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=word2idx['<PAD>'])
        self.transformer = nn.Transformer(d_model=self.d_model,
                                        encoder_layers=self.encoder_layers, 
                                        decoder_layers=self.decoder_layers,
                                        dim_feedforward=512,
                                        batch_first=True)

        # hidden_network用的原来的
        self.hidden_network = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size//2, self.output_size),
            nn.Tanh())

        self.positional_encoding = PositionalEncoding(self.d_model, dropout=0)

    def forward(self,src, tgt):
        # 没有分成批所以应该是单个序列跑的没有batch
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        src_key_padding_mask = transformer.get_key_padding_mask(src)
        tgt_key_padding_mask = transformer.get_key_padding_mask(tgt)

        # 对src和tgt进行编码
        # src = self.embedding(src)
        # tgt = self.embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 将准备好的数据送给transformer
        transformer_out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        hidden_network_out = self.hidden_network(transformer_out)
        chord_out = torch.sigmoid(hidden_network_out)
        return chord_out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask

    def Train(self):
        # 参数：train_loader是用来训练的数据集, verify是验证集, lr, Epoch数量
        # 论文提供的描述
        # We use minibatch gradient descent with categorical cross entropy as the cost function and Adam as the optimize
        train_loader = self.train_loader
        verify_loader = self.verify_loader
        lr = self.lr
        MAX_EPOCH = self.max_epoch

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(self.parameters(), lr)

        stop_cnt = 0
        pre_acc = 0
        for epoch in range(MAX_EPOCH):
            self.train()
            self.batch_size = self.train_loader.batch_size
            for melody, chord in train_loader:
                # melody=torch.tensor(melody).to(torch.float32)
                # chord=torch.tensor(chord).to(torch.float32)
                # melody = torch.tensor(melody)
                # chord = torch.tensor(chord)
                melody = melody.to(device)
                chord = chord.to(device)

                pred = self(melody)
                loss = criterion(pred, chord)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Epoch: ', '%04d' % (epoch + 1), 'loss = ', '{:.6f}'.format(loss))

            # verify, 输出一个平均正确率
            avg = 0
            cnt = 0
            self.batch_size = self.verify_loader.batch_size
            self.eval()
            for melody, chord in verify_loader:
                # melody=torch.tensor(melody).to(torch.float32)
                # chord=torch.tensor(chord).to(torch.float32)
                melody = melody.to(device)
                chord = chord.to(device)

                pred = self(melody)
                avg = avg + evaluation_simple(pred, chord, self.eval_config)  # 调用验证函数
                cnt = cnt + 1

                # if cnt % 100 == 0:
                #     print(f'count: {cnt}')

            print('Verify set average accuracy:', avg / cnt)
            print('')
            if (epoch > 1) & (pre_acc > avg / cnt):
                stop_cnt = stop_cnt + 1
                if stop_cnt >= self.early_stop:
                    break
            else:
                stop_cnt = 0
            pre_acc = avg / cnt
        torch.save(self, self.save_place)
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

            pred = self(melody)
            avg = avg + evaluation(pred, chord, self.eval_config)  # 调用验证函数
            cnt = cnt + 1

            # if cnt % 100 == 0:
            #     print(f'count: {cnt}')
        print('Test set average accuracy:', avg / cnt)
        return


def gen_model(train_loader, verify_loader, test_loader, config):
    '''
    args: train_loader, verify_loader, test_loader, config
    '''
    model=transformer(train_loader, verify_loader, test_loader, config)
    # model = torch.load('save/mymodel.pt')
    model.max_epoch = 1000
    print(model)
    return model.to(device)
