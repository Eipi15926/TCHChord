from torch.autograd import Variable
import torch.nn as nn

class MyLossFunc(nn.Module):
    def __init__(self):
        super(MyLossFunc, self).__init__()

    def forward(self, out, label):
        loss = 0
        lth = len(label)
        for i in (0,lth):
            out1 = out[i]
            label1 = label[i]
            loss =  loss + 1 - (out1 * label1).sum() /out1.sum()/label1.sum()
        loss = Variable(loss, requires_grad=True)
        return

# how to use:
'''
loss_function = MyLossFunc()
if cuda_available:
    loss_function.cuda()

for data in train_loader:  # 对于训练集的每一个batch
    img, label = data
    if cuda_available:
        img = img.cuda()
        label = label.cuda()

    out = cnn(img)  # 送进网络进行输出
    # print('out size: {}'.format(out.size()))
    # print('label size: {}'.format(label.size()))

    # out = torch.nn.functional.softmax(out, dim=1)
    loss = loss_function(out, label)  # 获得损失

    optimizer.zero_grad()  # 梯度归零
    loss.backward()  # 反向传播获得梯度，但是参数还没有更新
    optimizer.step()  # 更新梯度
'''