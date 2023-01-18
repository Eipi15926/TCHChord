import torch
'''
chord_list = []
# 生成和弦向量表
start = 9
for i in range(12):
    chord_major = [0,0,0,0,0,0,0,0,0,0,0,0]
    chord_minor = [0,0,0,0,0,0,0,0,0,0,0,0]
    chord_major[start] = 1
    chord_major[(start + 4) % 12] = 1
    chord_major[(start + 7) % 12] = 1
    chord_minor[start] = 1
    chord_minor[(start + 3) % 12] = 1
    chord_minor[(start + 7) % 12] = 1
    start = (start + 1) % 12
    chord_list.append(chord_major)
    chord_list.append(chord_minor)

chord_list = torch.tensor(chord_list)
'''
chord_list = torch.tensor([
    [0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
    [0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1.],
    [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
    [1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.],
    [1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.],
    [0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
    [0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
    [0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.],
    [0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.],
    [1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.],
    [1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.],
    [0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1.],
    [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0.],
    [1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1.]
])
# print(chord_list)

def color_process(color):  # 色差不在-6到6内
    if color > 6:
        color = color - 12
    elif color < -6:
        color = color + 12
    return color


def chord_color(chord):  # 和弦色彩
    value_list = torch.tensor([0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5])
    # print(torch.sum(value_list * chord))
    # print(torch.sum(chord))
    return torch.sum(value_list * chord) / torch.sum(chord)


def color_diff(chord_pre, chord_after):  # 和弦色差，默认三和弦
    lamda = torch.tensor(1 / 54)
    value_list = torch.tensor([0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5])
    color_pre = chord_color(chord_pre)
    color_after = chord_color(chord_after)
    color_diff = color_process(color_after - color_pre)
    chord_pre_new = torch.maximum(chord_pre - chord_after, torch.zeros(12))
    chord_after_new = torch.maximum(chord_after - chord_pre, torch.zeros(12))
    value = torch.tensor(0)
    for i in range(12):
        for j in range(12):
            if chord_pre_new[i] == 1 and chord_after_new[j] == 1:
                value = value + torch.abs(color_process(value_list[i] - value_list[j]))
    return 2 / 3.14 * torch.arctan(lamda * value)


# lim_num = 3
# lim = -1234567


def mapping(output_tensor, lim_num, lim):
    output = output_tensor.detach()
    # output = output_tensor
    x = torch.zeros(len(output))
    a = x
    for i in range(0,12):
        if output[i] > lim:
            x[i] = 1
        else:
            x[i] = 0
    b, idx1 = torch.sort(output, descending=True)
    index = idx1[:lim_num] # 前lim_sum大的数的索引
    a[index] = 1
    '''
    for i in range(0,12):
        if x[i] == 1:
            tmp = 0
            for j in range(0,lim_num):
                if i == index[j]:
                    tmp = 1
            x[i] = tmp
    '''
    x = a*x
    # print(x)
    dist_list = []
    if torch.sum(x) == 0:
        return 0
    for k in range(24):
        dist_list.append(color_diff(x, chord_list[k, :]))
    output_index = torch.argmin(torch.tensor(dist_list)) + 2
    # print(dist_list)
    return output_index

'''
def mapping2(output_tensor):  # 这里output为12的向量,返回对应和弦在one_hot编码中的位置
    #output = output_tensor.numpy()
    output = output_tensor
    shape = len(output)
    # 取最大三个值置1
    x = np.zeros(shape)
    index = np.argpartition(output, -3)[-3:]
    x[index] = 1
    output = x
    dist_list = []
    for j in range(24):
        dist_list.append(color_diff(output, chord_list[j, :]))
    output_index = np.argmin(dist_list) + 2
    return np.array(output_index)


Cmajor_chord = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
Gmajor_chord = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
Gseven_chord = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1])

output = np.array([-1, 3, 5, 1, 7, -4, 2, 8, 0, 3, 9, 0])
print(color_diff(Cmajor_chord, Gmajor_chord))
print(color_diff(Cmajor_chord, Gseven_chord))
print(mapping(output))
print(mapping2(output))
'''


# 输入每条旋律的两个12维向量的集合，输出该旋律的accuracy
def evaluation(ans_arr,label_arr, config):
    lim_num = config['lim_num']
    lim = config['lim']
    tot = len(label_arr)
    acc = 0
    for i in range(0,tot):
        ans_chord = mapping(ans_arr[i], lim_num, lim)
        label_chord = mapping(label_arr[i], lim_num, lim)
        if ans_chord == label_chord:
            acc = acc + 1
    return acc/tot
