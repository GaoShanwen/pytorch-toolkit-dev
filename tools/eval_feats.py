######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: eval_feats.py
# function: eval the accuracy of features in .npz-file.
######################################################
import argparse
import faiss
import numpy as np

import sys
sys.path.append('./')

from tools.post.write_mysql import save_keeps2mysql
from tools.post.feat_tools import load_data, run_compute, compute_acc_by_cat, choose_feats, print_acc_map, create_index, save_keeps_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', "--gallerys", type=str)
    parser.add_argument('-q', "--querys", type=str)
    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--save-sql', action='store_true', default=False)
    parser.add_argument("--num-classes", type=int, default=4091)
    parser.add_argument("--update-times", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def main(args):
    k = args.topk
    # 加载npz文件
    gallery_feature, gallery_label, gallery_files = load_data(args.gallerys)
    query_feature, query_label, _ = load_data(args.querys)
    faiss.normalize_L2(gallery_feature)
    faiss.normalize_L2(query_feature)
    cats = list(set(gallery_label))

    if args.debug:
        cat_index = np.where(gallery_label[:, np.newaxis] == cats[:10])[0]
        gallery_label = gallery_label[cat_index]
        gallery_feature = gallery_feature[cat_index]
        gallery_files = gallery_files[cat_index]

        cat_index = np.where(query_label[:, np.newaxis] == cats[:10])[0]
        query_label = query_label[cat_index]
        query_feature = query_feature[cat_index]
    
    # # print('index.ntotal=', index.ntotal, '\n')         # 输出index中包含的向量总数，为100000 
    index = create_index(gallery_feature, use_gpu=args.use_gpu)
    D, I = index.search(query_feature, k)
    p_label = gallery_label[I]
    with open(f'dataset/exp-data/removeredundancy/{args.num_classes}_cats.txt', 'r') as f: class_list = [line.strip('\n') for line in f.readlines()]
    acc_map = compute_acc_by_cat(p_label, query_label, class_list)
    print_acc_map(acc_map, 'eval_res.csv')
    run_compute(p_label, query_label, do_output=False)
    del index
    
    keeps = choose_feats(gallery_feature, gallery_label, samilar_thresh=args.threshold, use_gpu=False, update_times=args.update_times)
    print(f"original data: {gallery_label.shape[0]}, after remove samilar data: {keeps.shape[0]}")
    new_g_feats = gallery_feature[keeps]
    new_g_label = gallery_label[keeps]
    new_g_files = gallery_files[keeps]
    if args.save_sql:
        save_keeps2mysql(new_g_feats, new_g_label, new_g_files, class_list, update_times=args.update_times)
        return
    save_keeps_file(new_g_label, new_g_files, class_list, args.threshold)
    index = create_index(new_g_feats, use_gpu=args.use_gpu)

    # import pdb; pdb.set_trace()
    D, I = index.search(query_feature, k)
    p_label = new_g_label[I]
    acc_map = compute_acc_by_cat(p_label, query_label, class_list)
    print_acc_map(acc_map, f'eval_res-{args.threshold}.csv')
    run_compute(p_label, query_label, do_output=False)


if __name__ == '__main__':
    args = parse_args()
    # param, measure = 'Flat', faiss.METRIC_INNER_PRODUCT
    # param, measure = 'IVF100,Flat', faiss.METRIC_L2 # 倒排暴力检索：代表k-means聚类中心为100,
    # param, measure = 'IVF100,PQ16', faiss.METRIC_L2 # 倒排乘积量化：
    # param, measure = 'HNSW64', faiss.METRIC_L2      # HNSWx
    # param, measure = 'LSH', faiss.METRIC_L2           # 局部敏感哈希
    main(args)