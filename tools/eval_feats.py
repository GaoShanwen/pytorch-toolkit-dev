######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: eval_feats.py
# function: eval the accuracy of features in .npz-file.
######################################################
import argparse
import faiss

import pandas as pd

import sys
sys.path.append('./')

from tools.scripts.feat_tools import load_data, run_compute, compute_acc_by_cat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', "--gallerys", type=str)
    parser.add_argument('-q', "--querys", type=str)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    k = args.topk
    # 加载npz文件
    gallery_feature, gallery_label = load_data(args.gallerys)
    faiss.normalize_L2(gallery_feature)

    index = faiss.index_factory(args.dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.train(gallery_feature)
    index.add(gallery_feature)                           # 将向量库中的向量加入到index中
    # print('index.ntotal=', index.ntotal, '\n')         # 输出index中包含的向量总数，为100000 

    query_feature, query_label = load_data(args.querys)
    faiss.normalize_L2(query_feature)
    D, I = index.search(query_feature, k)
    p_label = gallery_label[I]
    acc_map = compute_acc_by_cat(p_label, query_label)
    df = pd.DataFrame(acc_map).transpose()
    df.to_csv('eval_res.csv')
    # print(df)
    # tp1_num, tp5_num = run_compute(p_label, query_label)
    # print(f"top1-knn(k={k}): {tp1_num}/{query_label.shape[0]}|{tp1_num/query_label.shape[0]}")
    # print(f"top5-knn(k={k}): {tp5_num}/{query_label.shape[0]}|{tp5_num/query_label.shape[0]}")
    
    # label_map = load_csv_file(args.label_file)
    # query_files = gallery_files[query_indics]
    # copy_files(p_label, query_label, label_map, query_files, "./dataset/exp-data/dns-all/error_rrwknn")
