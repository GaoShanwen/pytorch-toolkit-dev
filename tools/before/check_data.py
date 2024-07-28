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
from local_lib.utils.file_tools import save_dict2csv, load_csv_file

def load_data(anno_path):
    with open(anno_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip().split(',') for line in f.readlines()]# if line.startswith("/data/AI-scales/images")]
    filenames, labels = zip(*(lines))
    return filenames, labels


def check_data(filenames):
    for filename in filenames:#tqdm.tqdm(filenames):
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
    train_counter = collections.Counter(train_data).most_common()#[:999]
    val_dict = dict(collections.Counter(val_data))
    num_dict = {}
    for idx, (id, num1) in enumerate(train_counter):#.items():
        num2 = val_dict.get(id, '')
        id = id.replace(' ', '')
        cat = cat_map[id] if id in cat_map else cat_map[id[10:]]#''
        num_dict.update({idx: {"product_id": id, "name": cat, "train": num1, "val": num2}})
    save_dict2csv(num_dict, "data_static.csv")


if __name__=="__main__":
    data_root = "./dataset/optimize_task3" # "./dataset/optimize_24q3" #"dataset/function_test/need_1386" # 
    load_train_path = f"{data_root}/train.txt"
    # load_train_path = "train.txt"
    train_files, train_labels = load_data(load_train_path)
    load_val_path = f"{data_root}/val.txt"
    val_files, val_labels = load_data(load_val_path)
    # check_data(train_files)
    label_file = "./dataset/zero_dataset/label_names.csv"
    label_map = load_csv_file(label_file, frist_name=True)
    static_data(train_labels, val_labels, label_map)
