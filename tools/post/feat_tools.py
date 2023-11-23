######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: feat_tools.py
# function: the functions tools for feat extract or eval.
######################################################
import tqdm
import faiss
import numpy as np
import pandas as pd
from collections import Counter


def load_data(file_path):
    with np.load(file_path) as data:
        # 从npz文件中获取数组  
        feas = data['feats'].astype('float32')
        labels = data['gts']
        fpaths = data['fpaths']
    return feas, labels, fpaths


def run_compute(p_label, query_label, do_output=True, k=5):
    tp1_num = np.where(p_label[:, 0] == query_label)[0].shape[0]
    tp5_num = np.unique(np.where(p_label == query_label[:, np.newaxis])[0]).shape[0]
    if do_output:
        return tp1_num, tp5_num
    print(f"top1-knn(k={k}): {tp1_num}/{query_label.shape[0]}|{tp1_num/query_label.shape[0]}")
    print(f"top5-knn(k={k}): {tp5_num}/{query_label.shape[0]}|{tp5_num/query_label.shape[0]}")


def compute_acc_by_cat(p_label, query_label, class_list):
    top1_num, top5_num = run_compute(p_label, query_label)
    acc_map = {"all_data": {"top1_num": top1_num, "top1_acc": top1_num/query_label.shape[0], \
                            "top5_num": top5_num, "top5_acc": top5_num/query_label.shape[0], "data_num": query_label.shape[0]}}
    val_dict = dict(Counter(query_label))
    for cat, data_num in val_dict.items():
        cat_index = np.where(query_label == cat)
        top1_num = np.where(p_label[cat_index, 0] == query_label[cat_index])[1].shape[0]
        top5_num = np.unique(np.where(p_label[cat_index, :] == query_label[cat_index, np.newaxis])[1]).shape[0]
        acc_map[class_list[cat]] = {"top1_num": top1_num, "top1_acc": top1_num/data_num, "top5_num": top5_num, "top5_acc": top5_num/data_num, "data_num": data_num}
    return acc_map


def print_acc_map(acc_map, csv_name):
    df = pd.DataFrame(acc_map).transpose()
    df.to_csv(csv_name)
    # print(df)


def save_keeps_file(labels, files, class_list, threshold):
    obj_files = f"train-th{threshold}.txt"
    with open(obj_files, 'w') as f:
        for label_index, filename in zip(labels, files):
            label = class_list[label_index]
            f.write(f'{filename}, {label}\n')


def create_index(datas_embedding, use_gpu=False, param='Flat', measure=faiss.METRIC_INNER_PRODUCT):
    dim = datas_embedding.shape[1]
    index = faiss.index_factory(dim, param, measure)
    if use_gpu:
        index = faiss.index_cpu_to_gpus_list(index, gpus=[0, 1]) # gpus用于指定使用的gpu号
    index.train(datas_embedding)
    index.add(datas_embedding)   # 把向量数据加入索引
    return index


# 进行非极大值抑制
def choose_similarity(matrix, labels, samilar_thresh=0.9, use_gpu=False, update_times=0):
    cats = list(set(labels))
    stride = len(cats)//update_times if update_times else 1
    keeps = []
    pbar = tqdm.tqdm(total=len(cats), miniters=stride, maxinterval=600)
    for cat in cats:
        cat_index = np.where(labels == cat)[0]
        choose_gallery = matrix[cat_index]
        enable_gpu = use_gpu if choose_gallery.shape[0] <= 2048 else False
        index = create_index(choose_gallery, enable_gpu)
        g_g_dist, _ = index.search(choose_gallery, choose_gallery.shape[0])
        masks = []
        for i in range(cat_index.shape[0]-1):
            if i in masks: continue
            masks += np.where((g_g_dist[i, :] >= samilar_thresh) & (np.arange(cat_index.shape[0]) > i))[0].tolist()
        masks = np.array(masks)
        keeps += np.setdiff1d(cat_index, cat_index[masks]).tolist()
        pbar.update(1)
        del index
    pbar.close()
    return np.array(keeps)


def choose_noises(matrix, labels, choose_ratio, use_gpu=False, update_times=0):
    cats = list(set(labels))
    stride = len(cats)//update_times if update_times else 1
    keeps = []
    pbar = tqdm.tqdm(total=len(cats), miniters=stride, maxinterval=600)
    for cat in cats:
        cat_index = np.where(labels == cat)[0]
        choose_gallery = matrix[cat_index]
        enable_gpu = use_gpu if choose_gallery.shape[0] <= 2048 else False
        index = create_index(choose_gallery, enable_gpu)
        _, index_matric = index.search(choose_gallery, choose_gallery.shape[0])
        choose_num = int(cat_index.shape[0] * choose_ratio)
        last_count = Counter(index_matric[:, -choose_num:].reshape(-1).tolist()).most_common()
        keep = [key for key, _ in last_count][:choose_num]
        keeps += cat_index[np.array(keep, dtype=int)].tolist()
        pbar.update(1)
        del index
    pbar.close()
    return np.array(keeps)


def run_choose(matrix, labels, args):
    if args.remove_mode == "noise":
        return choose_noises(matrix, labels, args.threshold, use_gpu=args.use_gpu, update_times=args.update_times)
    elif args.remove_mode == "similarity":
        return choose_similarity(matrix, labels, args.threshold, use_gpu=args.use_gpu, update_times=args.update_times)
    else:
        raise f"{args.remove_mode} is not support!"