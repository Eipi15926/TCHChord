import torch
import torch.nn as nn
import torch.optim as optim
from metric import evaluation
from metric import evaluation_simple


# 一个用于修改target的函数，在训练集里target就是chord
def get_mask(mask):
    return mask


def func(output_p):
    output = output_p
    return output


def train(trmodel, device):
    train_loader = trmodel.train_loader
    verify_loader = trmodel.verify_loader
    lr = trmodel.lr
    MAX_EPOCH = trmodel.max_epoch

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(trmodel.parameters(), lr)

    for epoch in range(MAX_EPOCH):
        trmodel.batch_size = trmodel.train_loader.batch_size
        for melody, chord, target, mask in train_loader:
            lent = len(chord)
            output = []  # 应该是tensor 注意要改一下
            melody = melody.to(device)
            chord = chord.to(device)
            target = chord  # 对训练来说是这样的
            for i in range(lent):
                mask = get_mask(mask)
                output_p = trmodel(melody, target, mask)
                output = func(output_p)
            loss = criterion(output, chord)

            optimizer.zero_grad()
            trmodel.backward(loss)  # ??
            optimizer.step()


def test(trmodel, device):
    train_loader = trmodel.train_loader
    verify_loader = trmodel.verify_loader
    lr = trmodel.lr
    MAX_EPOCH = trmodel.max_epoch

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(trmodel.parameters(), lr)

    for epoch in range(MAX_EPOCH):
        trmodel.batch_size = trmodel.train_loader.batch_size
        for melody, chord, target, mask in train_loader:
            lent = len(chord)
            output = []  # 应该是tensor 注意要改一下
            melody = melody.to(device)
            chord = chord.to(device)
            target = chord  # 对训练来说是这样的
            for i in range(lent):
                mask = get_mask(mask)
                output_p = trmodel(melody, target, mask)
                output = func(output_p)
                target = output

            loss = criterion(output, chord)
            # 评估函数之后再改
            print("loss=", loss)
