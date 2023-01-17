<<<<<<< Updated upstream
import os
import random
from shutil import copy2


def data_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.1, test_scale=0.1):
    """
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹
    :param target_data_folder: 目标文件夹
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    """

    if train_scale + val_scale + test_scale != 1:
        return

    print("开始数据集划分")
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.makedirs(split_path)

    # 按照比例划分数据集，并进行数据的复制
    current_all_data = os.listdir(src_data_folder)
    current_data_length = len(current_all_data)
    current_data_index_list = list(range(current_data_length))
    random.shuffle(current_data_index_list)

    train_folder = os.path.join(target_data_folder, 'train')
    val_folder = os.path.join(target_data_folder, 'val')
    test_folder = os.path.join(target_data_folder, 'test')
    train_stop_flag = current_data_length * train_scale
    val_stop_flag = current_data_length * (train_scale + val_scale)
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    for i in current_data_index_list:
        src_img_path = os.path.join(src_data_folder, current_all_data[i])
        if current_idx <= train_stop_flag:
            copy2(src_img_path, train_folder)
            # print("{}复制到了{}".format(src_img_path, train_folder))
            train_num = train_num + 1
        elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            copy2(src_img_path, val_folder)
            # print("{}复制到了{}".format(src_img_path, val_folder))
            val_num = val_num + 1
        else:
            copy2(src_img_path, test_folder)
            # print("{}复制到了{}".format(src_img_path, test_folder))
            test_num = test_num + 1
        current_idx = current_idx + 1

    print("数据集{}按照{}：{}：{}的比例划分完成，一共{}个文件".format(src_data_folder, train_scale, val_scale, test_scale, current_data_length))
    print("训练集{}：{}个文件".format(train_folder, train_num))
    print("验证集{}：{}个文件".format(val_folder, val_num))
    print("测试集{}：{}个文件".format(test_folder, test_num))


if __name__ == '__main__':
    src_data_folder = "..\\Data\\all_key"
    target_data_folder = "..\\Data\\classify"
    data_split(src_data_folder, target_data_folder)
=======
import os
import random
from shutil import copy2


def data_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.1, test_scale=0.1):
    """
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹
    :param target_data_folder: 目标文件夹
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    """

    if train_scale + val_scale + test_scale != 1:
        return

    print("开始数据集划分")
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.makedirs(split_path)

    # 按照比例划分数据集，并进行数据的复制
    current_all_data = os.listdir(src_data_folder)
    current_data_length = len(current_all_data)
    current_data_index_list = list(range(current_data_length))
    random.shuffle(current_data_index_list)

    train_folder = os.path.join(target_data_folder, 'train')
    val_folder = os.path.join(target_data_folder, 'val')
    test_folder = os.path.join(target_data_folder, 'test')
    train_stop_flag = current_data_length * train_scale
    val_stop_flag = current_data_length * (train_scale + val_scale)
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    for i in current_data_index_list:
        src_img_path = os.path.join(src_data_folder, current_all_data[i])
        if current_idx <= train_stop_flag:
            copy2(src_img_path, train_folder)
            # print("{}复制到了{}".format(src_img_path, train_folder))
            train_num = train_num + 1
        elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            copy2(src_img_path, val_folder)
            # print("{}复制到了{}".format(src_img_path, val_folder))
            val_num = val_num + 1
        else:
            copy2(src_img_path, test_folder)
            # print("{}复制到了{}".format(src_img_path, test_folder))
            test_num = test_num + 1
        current_idx = current_idx + 1

    print("数据集{}按照{}：{}：{}的比例划分完成，一共{}个文件".format(src_data_folder, train_scale, val_scale, test_scale, current_data_length))
    print("训练集{}：{}个文件".format(train_folder, train_num))
    print("验证集{}：{}个文件".format(val_folder, val_num))
    print("测试集{}：{}个文件".format(test_folder, test_num))


if __name__ == '__main__':
    src_data_folder = "..\\Data\\all_key"
    target_data_folder = "..\\Data\\classify"
    data_split(src_data_folder, target_data_folder)
>>>>>>> Stashed changes
