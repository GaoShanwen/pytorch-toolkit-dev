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
from tools.post.feat_tools import load_data, run_compute, compute_acc_by_cat, run_choose, print_acc_map, create_index, save_keeps_file
from tools.post.knn import get_predict_label
from tools.visualize.vis_error import save_imgs, load_csv_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', "--gallerys", type=str)
    parser.add_argument('-q', "--querys", type=str)
    parser.add_argument("--save-root", type=str, default="output/vis/noises")
    parser.add_argument("--label-file", type=str,
                        default='dataset/exp-data/zero_dataset/label_names.csv')
    parser.add_argument("--cats-file", type=str, 
                        default='dataset/exp-data/removeredundancy/629_cats.txt')
    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--use-knn', action='store_true', default=False)
    parser.add_argument('--use-sgd', action='store_true', default=False)
    parser.add_argument('--remove-mode', type=str, default="none", 
                        help="remove mode (eq: none, noise, or similarity)!")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--save-sql', action='store_true', default=False)
    parser.add_argument("--num-classes", type=int, default=629)
    parser.add_argument("--update-times", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def run_eval(g_feats, g_label, q_feats, q_label, class_list, args, acc_file_name='eval_res.csv'):
    index = create_index(g_feats, use_gpu=args.use_gpu, param=args.param, measure=args.measure)
    D, I = index.search(q_feats, args.topk)
    p_label = get_predict_label(D, I, g_label, q_label, use_knn=args.use_knn, use_sgd=args.use_sgd)
    acc_map = compute_acc_by_cat(p_label, q_label, class_list)
    if acc_file_name: print_acc_map(acc_map, acc_file_name)
    run_compute(p_label, q_label, do_output=False)


def main(args):
    # 加载npz文件
    gallery_feature, gallery_label, gallery_files = load_data(args.gallerys)
    query_feature, query_label, _ = load_data(args.querys)
    faiss.normalize_L2(gallery_feature)
    faiss.normalize_L2(query_feature)

    if args.debug:
        cats = list(set(gallery_label))
        cat_index = np.where(gallery_label[:, np.newaxis] == cats[:15])[0]
        gallery_label, gallery_feature, gallery_files = gallery_label[cat_index], gallery_feature[cat_index], gallery_files[cat_index]

        cat_index = np.where(query_label[:, np.newaxis] == cats[:15])[0]
        query_label, query_feature = query_label[cat_index], query_feature[cat_index]
    
    with open(args.cats_file, 'r') as f: class_list = [line.strip('\n')[1:] for line in f.readlines()]
    run_eval(gallery_feature, gallery_label, query_feature, query_label, class_list, args, 'eval_res.csv')
    # print('index.ntotal=', index.ntotal, '\n')         # 输出index中包含的向量总数，为100000 

    if args.remove_mode == "none": 
        if args.save_sql:
            save_keeps2mysql(gallery_feature, gallery_label, gallery_files, class_list, update_times=args.update_times)
        return
    keeps = run_choose(gallery_feature, gallery_label, args)
    if args.remove_mode == "noise":
        # label_index = load_csv_file(args.label_file)
        # cats = list(set(gallery_label))
        # label_map = {i: label_index[cat].split('/')[0] for i, cat in enumerate(class_list) if i in cats}
        # save_imgs(new_g_label, new_g_files, label_map, args.save_root)

        all_indics = np.arange(gallery_label.shape[0])
        keeps = np.setdiff1d(all_indics, keeps)
        # in_set = np.in1d(all_indics, keeps)
        # keeps = all_indics[~in_set]
        # new_g_feats, new_g_label, new_g_files = gallery_feature[keeps], gallery_label[keeps], gallery_files[keeps]
        # run_eval(new_g_feats, new_g_label, query_feature, query_label, class_list, args, '')
        # return
    
    print(f"original data: {gallery_label.shape[0]}, after remove {args.remove_mode} data: {keeps.shape[0]}")
    new_g_feats, new_g_label, new_g_files = gallery_feature[keeps], gallery_label[keeps], gallery_files[keeps]
    np.savez(f'output/feats/regnety_040-train-{args.threshold}.npz', feats=new_g_feats, gts=new_g_label, fpaths=new_g_files)
    if args.save_sql:
        save_keeps2mysql(new_g_feats, new_g_label, new_g_files, class_list, update_times=args.update_times)
    # save_keeps_file(new_g_label, new_g_files, class_list, args.threshold)
    csv_name = '' #f'eval_res-{args.remove_mode}-{args.threshold}.csv'
    run_eval(new_g_feats, new_g_label, query_feature, query_label, class_list, args, csv_name)


if __name__ == '__main__':
    args = parse_args()
    args.param = f'IVF{args.num_classes},Flat'
    args.measure = faiss.METRIC_INNER_PRODUCT
    # param, measure = 'Flat', faiss.METRIC_INNER_PRODUCT
    # param, measure = 'IVF100,Flat', faiss.METRIC_L2 # 倒排暴力检索：代表k-means聚类中心为100,
    # param, measure = 'IVF100,PQ16', faiss.METRIC_L2 # 倒排乘积量化：
    # param, measure = 'HNSW64', faiss.METRIC_L2      # HNSWx
    # param, measure = 'LSH', faiss.METRIC_L2           # 局部敏感哈希
    main(args)