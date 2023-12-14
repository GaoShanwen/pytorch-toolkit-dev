######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: check_data.py
# function: check custom data before training.
######################################################
import os
import collections


def load_data(anno_path):
    with open(anno_path, "r") as f:
        lines = [line.strip().split(",") for line in f.readlines()]  # if line.startswith("/data/AI-scales/images")
    filenames, labels = zip(*(lines))
    return filenames, labels


def load_csv_file(label_file):
    product_id_map = {}
    with open(label_file) as f:
        for line in f:
            # id_record = line.strip().split()
            try:
                id_record = line.strip().replace("'", "").replace('"', "").split(",")
                product_id_map[id_record[0]] = id_record[1]
            except:
                import pdb

                pdb.set_trace()
    return product_id_map


def check_data(filenames):
    for filename in filenames:
        if not os.path.exists(filename):
            print(filename)
        try:
            with open(filename, "rb") as f:
                f.seek(-2, 2)
                if not f.read() == b"\xff\xd9":
                    print(filename)
        except IOError:
            print(filename)


def static_data(train_data, val_data, cat_map):
    # save_cats, _ = zip(*collections.Counter(train_data).most_common())#[:4281])#[:3000])
    # # print(len([id for id, _ in collections.Counter(val_data).items() if id in save_cats]))
    # for cat in save_cats:
    #     print(cat)
    train_counter = collections.Counter(train_data).most_common()  # [:999]
    val_dict = dict(collections.Counter(val_data))
    # train_dict = dict(collections.Counter(train_data))
    # val_counter = collections.Counter(val_data).most_common()
    check_dict = val_dict  # train_dict
    show_counter = train_counter  # val_counter

    print("| =- cat id -= |  ====----    n a m e    ----====  | =- train -= |  =- val -=  |")
    # train_counter = []
    for id, num1 in show_counter:  # .items():
        num2 = check_dict.get(id, "")
        # if num1 <= 30:# and isinstance(num2, str):
        #     continue
        id = id.replace(" ", "")
        cat = cat_map[id].split("/")[0]
        print(f"| {id: ^12} | {cat: ^30} | {num1: ^10} | {num2: ^10} |")
    # keys = [name for name in val_dict.keys() if name in train_counter]
    # print(len(keys))


if __name__ == "__main__":
    load_train_path = "./dataset/blacklist2/train.txt"
    img_root = "dataset/blacklist2"
    with open(load_train_path, "w") as write_f:
        for folder in sorted(os.listdir(img_root)):
            if folder.endswith(".txt"):
                continue
            folder_dir = os.path.join(img_root, folder)
            for img_name in os.listdir(folder_dir):
                img_path = os.path.join(folder_dir, img_name)
                write_f.writelines(f"{img_path}, {folder}\n")
    train_files, train_labels = load_data(load_train_path)

    # # load_train_path = "./dataset/zero_dataset/train.txt"
    # root_dir = "dataset/zero_600w"
    # load_train_path = os.path.join(root_dir, "train.txt")
    # train_files, train_labels = load_data(load_train_path)
    # # check_data(train_files)
    # # load_val_path = "./dataset/zero_dataset/val.txt"
    # load_val_path = os.path.join(root_dir, "val.txt")
    # val_files, val_labels = load_data(load_val_path)
    # # check_data(val_files)
    # label_file = "./dataset/zero_dataset/label_names.csv"
    # label_map = load_csv_file(label_file)
    # static_data(train_labels, val_labels, label_map)
