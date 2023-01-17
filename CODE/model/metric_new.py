import numpy as np

chord_list = []

# 生成和弦向量表
start = 9
for i in range(12):
    chord_major = np.zeros(12)
    chord_minor = np.zeros(12)
    chord_major[start] = 1
    chord_major[(start + 4) % 12] = 1
    chord_major[(start + 7) % 12] = 1
    chord_minor[start] = 1
    chord_minor[(start + 3) % 12] = 1
    chord_minor[(start + 7) % 12] = 1
    start = (start + 1) % 12
    chord_list.append(chord_major)
    chord_list.append(chord_minor)

chord_list = np.array(chord_list)


# print(chord_list)

def color_process(color):  # 色差不在-6到6内
    if color > 6:
        color = color - 12
    elif color < -6:
        color = color + 12
    return color


def chord_color(chord):  # 和弦色彩
    value_list = np.array([0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5])
    return np.sum(value_list * chord) / np.sum(chord)


def color_diff(chord_pre, chord_after):  # 和弦色差，默认三和弦
    lamda = 1 / 54
    value_list = np.array([0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5])
    color_pre = chord_color(chord_pre)
    color_after = chord_color(chord_after)
    color_diff = color_process(color_after - color_pre)
    chord_pre_new = np.maximum(chord_pre - chord_after, 0)
    chord_after_new = np.maximum(chord_after - chord_pre, 0)
    value = 0
    for i in range(12):
        for j in range(12):
            if chord_pre_new[i] == 1 and chord_after_new[j] == 1:
                value = value + np.abs(color_process(value_list[i] - value_list[j]))
    return 2 / 3.14 * np.arctan(lamda * value)


def mapping(output_tensor):  # 这里output为12的向量,返回对应和弦在one_hot编码中的位置
    output = output_tensor.detach().numpy()
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


'''
Cmajor_chord = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
Gmajor_chord = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
Gseven_chord = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1])

output = np.array([-1, 3, 5, 1, 7, -4, 2, 8, 0, 3, 9, 0])
print(color_diff(Cmajor_chord, Gmajor_chord))
print(color_diff(Cmajor_chord, Gseven_chord))
print(mapping(output))
'''


# 输入每条旋律的两个12维向量的集合，输出该旋律的accuracy
def evaluation(ans_arr,label_arr):
    tot = len(label_arr)
    acc = 0
    for i in range(0,tot):
        ans_chord = mapping(ans_arr[i])
        label_chord = mapping(label_arr[i])
        if ans_chord == label_chord:
            acc = acc + 1
    return acc/tot
