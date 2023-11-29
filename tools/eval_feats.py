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
# from tools.post.feat_tools import load_data, run_compute, compute_acc_by_cat, run_choose, print_acc_map, create_index, save_keeps_file, get_predict_label
from tools.post import feat_tools
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
    parser.add_argument("--pass-cats", type=str, 
                        default='dataset/exp-data/removeredundancy/pass_cats.txt')
    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--use-knn', action='store_true', default=False)
    parser.add_argument('--use-sgd', action='store_true', default=False)
    parser.add_argument('--run-test', action='store_true', default=False)
    parser.add_argument('--save-detail', action='store_true', default=False)
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
    index = feat_tools.create_index(g_feats, use_gpu=args.use_gpu, param=args.param, measure=args.measure)
    D, I = index.search(q_feats, args.topk)
    p_label = feat_tools.get_predict_label(D, I, g_label, q_label, use_knn=args.use_knn, use_sgd=args.use_sgd)
    if acc_file_name:
        label_index = load_csv_file(args.label_file)
        cats = list(set(g_label))
        label_map = {i: label_index[cat].split('/')[0] for i, cat in enumerate(class_list) if i in cats}
        acc_map = feat_tools.compute_acc_by_cat(p_label, q_label, class_list, label_map)
        feat_tools.print_acc_map(acc_map, acc_file_name)
    feat_tools.run_compute(p_label, q_label, do_output=False)


def main(gallery_feature, gallery_label, gallery_files, query_feature, query_label, class_list, args):
    run_eval(gallery_feature, gallery_label, query_feature, query_label, class_list, args, 'eval_res.csv')

    if args.remove_mode == "none":
        if args.save_sql:
            save_keeps2mysql(gallery_feature, gallery_label, gallery_files, class_list, update_times=args.update_times)
        return
    keeps = feat_tools.run_choose(gallery_feature, gallery_label, args)
    if args.remove_mode == "noise":
        if args.save_detail:
            label_index = load_csv_file(args.label_file)
            cats = list(set(gallery_label))
            label_map = {i: label_index[cat].split('/')[0] for i, cat in enumerate(class_list) if i in cats}
            save_imgs(new_g_label, new_g_files, label_map, args.save_root)
        all_indics = np.arange(gallery_label.shape[0])
        keeps = np.setdiff1d(all_indics, keeps)
    
    print(f"original data: {gallery_label.shape[0]}, after remove {args.remove_mode} data: {keeps.shape[0]}")
    new_g_feats, new_g_label, new_g_files = gallery_feature[keeps], gallery_label[keeps], gallery_files[keeps]
    np.savez(f'output/feats/regnety_040-train-{args.threshold}.npz', feats=new_g_feats, gts=new_g_label, fpaths=new_g_files)
    if args.save_sql:
        save_keeps2mysql(new_g_feats, new_g_label, new_g_files, class_list, update_times=args.update_times)
    feat_tools.save_keeps_file(new_g_label, new_g_files, class_list, f"train-th{args.threshold}.txt")
    csv_name = f'eval_res-{args.remove_mode}-{args.threshold}.csv' if args.save_detail else ''
    run_eval(new_g_feats, new_g_label, query_feature, query_label, class_list, args, csv_name)


def run_test(g_feats, g_label, q_feats, q_labels, q_files, class_list, args):
    static = feat_tools.run_choose(g_feats, g_label, args)
    label_index = load_csv_file(args.label_file)
    cats = list(set(g_label))
    label_map = {i: label_index[cat].split('/')[0] for i, cat in enumerate(class_list) if i in cats}
    new_static = {}
    for cat, value in static.items():
        new_value = {}
        for k, v in value.items():
            value_name = label_map[v] if k.endswith("name") else v
            new_value.update({k: value_name})
        new_value.update({"id": class_list[cat]})
        new_static.update({label_map[cat]: new_value})
    feat_tools.print_acc_map(new_static, "static.csv")
    # import pdb; pdb.set_trace()
    # #################### test knn ####################
    # index = feat_tools.create_index(g_feats, use_gpu=args.use_gpu, param=args.param, measure=args.measure)
    # D, I = index.search(q_feats, args.topk)
    # p_label = feat_tools.get_predict_label(D, I, g_label, q_labels, use_knn=False)
    # original_errors = np.where(p_label[:, 0] != q_labels)[0]
    # p_label = feat_tools.get_predict_label(D, I, g_label, q_labels, use_knn=True)
    # knn_errors = np.where(p_label[:, 0] != q_labels)[0]

    # e_only_in_knn = np.setdiff1d(knn_errors, original_errors)
    # print(f"new-errors knn-k={args.topk}: {e_only_in_knn.shape[0]} / {q_labels.shape[0]} | " +
    #       f"({original_errors.shape[0]} -> {knn_errors.shape[0]})")
    # new_q_labels, new_q_files = q_labels[e_only_in_knn], q_files[e_only_in_knn]
    # feat_tools.save_keeps_file(new_q_labels, new_q_files, class_list, f"new_errors-knn.txt")


if __name__ == '__main__':
    args = parse_args()
    # args.param = f'IVF{args.num_classes},Flat'
    args.param = f'IVF400,Flat'
    args.measure = faiss.METRIC_INNER_PRODUCT
    # args.param, args.measure = 'Flat', faiss.METRIC_INNER_PRODUCT
    # param, measure = 'IVF100,Flat', faiss.METRIC_L2 # 倒排暴力检索：代表k-means聚类中心为100,
    # param, measure = 'IVF100,PQ16', faiss.METRIC_L2 # 倒排乘积量化：
    # param, measure = 'HNSW64', faiss.METRIC_L2      # HNSWx
    # param, measure = 'LSH', faiss.METRIC_L2           # 局部敏感哈希

    # 加载npz文件
    g_feature, g_label, g_files = feat_tools.load_data(args.gallerys)
    q_feature, q_label, q_files = feat_tools.load_data(args.querys)
    faiss.normalize_L2(g_feature)
    faiss.normalize_L2(q_feature)

    with open(args.cats_file, 'r') as f: class_list = [line.strip('\n')[1:] for line in f.readlines()]
    if args.debug:
        with open(args.pass_cats, 'r') as f: mask_cats = [line.strip('\n') for line in f.readlines()]
        choose_cats = list(set(g_label))
        # print(class_list, len(choose_cats))
        for cat in mask_cats:
            choose_cats.remove(class_list.index(cat))
        # choose_cats = list(set(g_label))[:15]
        cat_index = np.where(g_label[:, np.newaxis] == choose_cats)[0]
        g_feature, g_label, g_files = g_feature[cat_index], g_label[cat_index], g_files[cat_index]

        cat_index = np.where(q_label[:, np.newaxis] == choose_cats)[0]
        q_feature, q_label, q_files =  q_feature[cat_index], q_label[cat_index], q_files[cat_index]
    
    if args.run_test:
        run_test(g_feature, g_label, q_feature, q_label, q_files, class_list, args)
    else:
        main(g_feature, g_label, g_files, q_feature, q_label, class_list, args)