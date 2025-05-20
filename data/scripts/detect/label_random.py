import os
# from os import listdir, getcwd
# from os.path import join
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser("visualize boxes parameters")
    # parser.add_argument("-s", "--src-files", type=str, default=None)
    # parser.add_argument("-o", "--obj-root", type=str, default="vis_imgs")
    parser.add_argument("-t", "--task", type=str, required=True, default=None)
    # parser.add_argument("--k", type=int, default=1)
    return parser.parse_args()


#训练集+验证集比例 小于等于1
#训练集比例
double_train_property = 0.8
#验证集比例
double_val_property = 0.1

args = parse_args()
data_dir = f"data/det-dataset/{args.task}"
f=open(os.path.join(data_dir, 'total_train.txt'), 'w')
# 遍历数据路径dataset_dir
def traversal_dataset_dir(dateset_dir_txt):
    with open(dateset_dir_txt, 'r') as file:
        for path in file:
            path = path.rstrip('\n')
            if os.path.isdir(path):
                traversal_imgs(path)
            else:
                print("路径不存在：",path)


# 遍历路径下图片写入到total_train.txt
def traversal_imgs(inputdir):
    for dirpath, dirnames, filenames in os.walk(inputdir):
        for name in filenames:
            # if 'jpg' in name:
            if name.split(".")[-1] in ["jpg", "png", "jpeg"]:
                f.write(os.path.join(dirpath, name)+'\n')


#将total_train.txt分为训练集、验证集、测试集              
def get_all_imgs_dir(input_txt):
    with open(input_txt, 'r') as file:
        lines = file.readlines()
        #print(lines)
        #print("----------------------------------")
        random.shuffle(lines)
        #print(lines)
        total_lines = len(lines)
        train_data = int(total_lines * double_train_property)
        val_data = int(total_lines * double_val_property)
        test_data = total_lines - train_data - val_data
        with open(os.path.join(data_dir, 'train.txt'), 'w') as file:
            file.writelines(lines[:train_data])
        with open(os.path.join(data_dir, 'val.txt'), 'w') as file:
            file.writelines(lines[train_data:train_data + val_data])
        with open(os.path.join(data_dir, 'test.txt'), 'w') as file:
            file.writelines(lines[train_data + val_data:])
        print("总数据：%d"%total_lines)
        print("训练集：%d"%train_data)
        print("验证集：%d"%val_data)
        print("测试集：%d"%test_data)
        
traversal_dataset_dir(os.path.join(data_dir, 'dataset_dir.txt'))
f.close()
get_all_imgs_dir(os.path.join(data_dir, 'total_train.txt'))
print("complete")
