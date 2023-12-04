######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: ready_det_data.py
# function: copy n-picture from every categoies in
#     owner data for labeling detection data.
######################################################
import os
import tqdm
import numpy as np
import cv2
import shutil
import collections


def load_data(anno_path):
    with open(anno_path, "r") as f:
        lines = [line.strip().split(",") for line in f.readlines() if line.startswith("/data/AI-scales/images")]
    filenames, labels = zip(*(lines))
    return filenames, labels


def copy_choose_data(obj_dir, cat_file, anno_path, choose_num=5, num_classes=4281):
    with open(cat_file, "r") as f:
        save_cats = [line.strip("\n") for line in f.readlines()]
    choose_cats = save_cats[:num_classes] if num_classes < len(save_cats) else save_cats
    train_files, train_labels = load_data(anno_path)
    static = dict(collections.Counter(train_labels).most_common())  #
    for key, value in static.items():
        if key not in choose_cats:
            continue
        static[key] = min(value, choose_num)

    label_file = f"{obj_dir}-001.txt"
    with open(label_file, "w") as f:
        for filename, label in zip(train_files, train_labels):  # tqdm.tqdm():
            if label in choose_cats:
                shutil.copy(filename, obj_dir)
                static[label] -= 1
                f.writelines(f"{filename},{label}\n")
                if not static[label]:
                    choose_cats.remove(label)
            if not len(choose_cats):
                return
    # basename = filename.split('/')[-1]
    # if basename in histories:
    #     print(cats[histories.index(basename)], label)
    # else:
    #     histories.append(basename)
    #     cats.append(label)
    # print(filename, label)
    # print(static)


if __name__ == "__main__":
    cat_file = "./dataset/zero_dataset/save_cats.txt"
    anno_path = "dataset/zero_dataset/train.txt"
    obj_dir = "./dataset/minidata/detection"
    copy_choose_data(obj_dir, cat_file, anno_path, choose_num=1)
