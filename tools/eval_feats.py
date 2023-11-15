######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: eval_feats.py
# function: eval the accuracy of features in .npz-file.
######################################################
import argparse
import faiss
import tqdm
import numpy as np
import pandas as pd

import sys
sys.path.append('./')

from tools.scripts.feat_tools import load_data, run_compute, compute_acc_by_cat, choose_feats, print_acc_map, create_index, save_keeps_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', "--gallerys", type=str)
    parser.add_argument('-q', "--querys", type=str)
    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    k = args.topk
    param, measure = 'Flat', faiss.METRIC_INNER_PRODUCT
    # param, measure = 'IVF100,Flat', faiss.METRIC_L2 # 倒排暴力检索：代表k-means聚类中心为100,
    # param, measure = 'IVF100,PQ16', faiss.METRIC_L2 # 倒排乘积量化：
    # param, measure = 'HNSW64', faiss.METRIC_L2      # HNSWx
    # param, measure = 'LSH', faiss.METRIC_L2           # 局部敏感哈希
    # 加载npz文件
    gallery_feature, gallery_label, gallery_files = load_data(args.gallerys)
    faiss.normalize_L2(gallery_feature)
    cats = list(set(gallery_label))

    if args.debug:
        cat_index = np.where(gallery_label[:, np.newaxis] == cats[:10])
        gallery_label = gallery_label[cat_index[0]]
        gallery_feature = gallery_feature[cat_index[0]]
        gallery_files = gallery_files[cat_index[0]]
    # # print('index.ntotal=', index.ntotal, '\n')         # 输出index中包含的向量总数，为100000 
    index = create_index(gallery_feature, use_gpu=args.use_gpu)
    query_feature, query_label, _ = load_data(args.querys)
    if args.debug:
        cat_index = np.where(query_label[:, np.newaxis] == cats[:10])
        query_label = query_label[cat_index[0]]
        query_feature = query_feature[cat_index[0]]
    faiss.normalize_L2(query_feature)
    D, I = index.search(query_feature, k)
    p_label = gallery_label[I]
    with open('dataset/exp-data/removeredundancy/4091_cats.txt', 'r') as f: class_list = [line.strip('\n') for line in f.readlines()]
    acc_map = compute_acc_by_cat(p_label, query_label, class_list)
    print_acc_map(acc_map, 'eval_res.csv')
    tp1_num, tp5_num = run_compute(p_label, query_label)
    print(f"top1-knn(k={k}): {tp1_num}/{query_label.shape[0]}|{tp1_num/query_label.shape[0]}")
    print(f"top5-knn(k={k}): {tp5_num}/{query_label.shape[0]}|{tp5_num/query_label.shape[0]}")
    del index
    
    keeps = choose_feats(gallery_feature, gallery_label, samilar_thresh=args.threshold, use_gpu=False)
    save_keeps_file(gallery_files, gallery_label, gallery_label[keeps], class_list, args.threshold)
    new_g_feats = gallery_feature[keeps]
    new_g_label = gallery_label[keeps]
    index = create_index(new_g_feats, use_gpu=args.use_gpu)

    D, I = index.search(query_feature, k)
    p_label = new_g_label[I]
    acc_map = compute_acc_by_cat(p_label, query_label, class_list)
    print_acc_map(acc_map, f'eval_res-{args.threshold}.csv')
    tp1_num, tp5_num = run_compute(p_label, query_label)
    print(f"original data: {gallery_label.shape[0]}, after remove samilar data: {new_g_label.shape[0]}")
    print(f"top1-knn(k={k}): {tp1_num}/{query_label.shape[0]}|{tp1_num/query_label.shape[0]}")
    print(f"top5-knn(k={k}): {tp5_num}/{query_label.shape[0]}|{tp5_num/query_label.shape[0]}")

