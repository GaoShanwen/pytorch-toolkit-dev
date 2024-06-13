######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: check_data.py
# function: check owner data before training.
######################################################
import os 
import tqdm
from PIL import Image
import collections


def load_data(anno_path):
    with open(anno_path, 'r') as f:
        lines = [line.strip().split(',') for line in f.readlines() if line.startswith("/data/AI-scales/images")]
    filenames, labels = zip(*(lines))
    return filenames, labels


def check_data(filenames):
    for filename in tqdm.tqdm(filenames):
        if not os.path.exists(filename):
            print(filename)
        try:  
            # Image.open(filename).verify()  
            with open(filename, 'rb') as f:
                f.seek(-2, 2)
                if not f.read() == b'\xff\xd9':
                    print(filename)
        except IOError:  
            print(filename)


def static_data(train_data, val_data, cat_map):
    # save_cats, _ = zip(*collections.Counter(train_data).most_common()[:4281])#[:3000])
    # print(len([id for id, _ in collections.Counter(val_data).items() if id in save_cats]))
    # # for cat in save_cats:
    # #     print(cat)
    train_counter = collections.Counter(train_data).most_common()#[:999]
    val_dict = dict(collections.Counter(val_data))
    # train_dict = dict(collections.Counter(train_data))
    # val_counter = collections.Counter(val_data).most_common()
    check_dict = val_dict #train_dict
    show_counter = train_counter #val_counter
    
    # import pdb; pdb.set_trace()
    print("| =- cat id -= |  ====----    n a m e    ----====  | =- train -= |  =- val -=  |")
    # train_counter = []
    for id, num1 in show_counter:#.items():
        num2 = check_dict.get(id, '')
        id = id.replace(' ', '')
        cat = cat_map[id].split('/')[0]
        print(f"| {id: ^12} | {cat: ^30} | {num1: ^10} | {num2: ^10} |")
    # keys = [name for name in val_dict.keys() if name in train_counter]
    # print(len(keys))


if __name__=="__main__":
    load_train_path = "./dataset/exp-data/zero_dataset/train.txt"
    train_files, train_labels = load_data(load_train_path)
    load_val_path = "./dataset/exp-data/zero_dataset/val.txt"
    val_files, val_labels = load_data(load_val_path)
    # check_data(filenames)
    label_file = "./dataset/exp-data/zero_dataset/label_names.csv"
    label_map = load_csv_file(label_file)
    static_data(train_labels, val_labels, label_map)
    # num = 9
    # print(f"{num:03d}")
