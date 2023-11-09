######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: feat_tools.py
# function: the functions tools for feat extract or eval.
######################################################
import tqdm
import numpy as np
from collections import Counter


def load_data(file_path):
    with np.load(file_path) as data:
        # 从npz文件中获取数组  
        feas = data['feats'].astype('float32')
        labels = data['gts']
    return feas, labels


def run_compute(p_label, query_label):
    tp1_num, tp5_num = 0, 0
    for (pred, gt) in zip(p_label, query_label):
        tp5_num += gt in pred
        tp1_num += gt == pred[0]
    # tp_num = np.where(p_label == query_label)[0].shape[0]
    return tp1_num, tp5_num


def compute_acc_by_cat(p_label, query_label):
    top1_num, top5_num = run_compute(p_label, query_label)
    acc_map = {"all_data": {"top1_num": top1_num, "top1_acc": top1_num/query_label.shape[0], \
                            "top5_num": top5_num, "top5_acc": top5_num/query_label.shape[0], "data_num": query_label.shape[0]}}
    val_dict = dict(Counter(query_label))
    for cat, data_num in tqdm.tqdm(val_dict.items()):
        cat_index = np.where(query_label == cat)
        top1_num = np.where(p_label[cat_index, 0] == query_label[cat_index])[1].shape[0]
        top5_num = len(set(np.where(p_label[cat_index, :] == query_label[cat_index, np.newaxis])[1]))
        acc_map[cat] = {"top1_num": top1_num, "top1_acc": top1_num/data_num, "top5_num": top5_num, "top5_acc": top5_num/data_num, "data_num": data_num}
    # import pdb; pdb.set_trace()
    return acc_map
