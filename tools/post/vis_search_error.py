######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.16
# filenaem: vis_search_error.py
# function: visualize the error picture for search results.
######################################################
import argparse

import faiss
import numpy as np

from local_lib.utils.feat_tools import create_index
from local_lib.utils.file_tools import load_csv_file, load_data
from local_lib.utils.visualize import VisualizeResults


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gallerys", type=str)
    parser.add_argument("-q", "--querys", type=str)
    parser.add_argument("--label-file", type=str, default="dataset/zero_dataset/label_names.csv")
    parser.add_argument("--cats-file", type=str, default="dataset/removeredundancy/629_cats.txt")
    parser.add_argument("--pass-cats", type=str, default="dataset/removeredundancy/pass_cats.txt")
    parser.add_argument("--mask-path", type=str, default="blacklist-val.npy")
    parser.add_argument("--pass-remove", action="store_true", default=False)
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--save-root", type=str, default="output/vis/errors")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--text-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=9)
    return parser.parse_args()


def vis_error(g_feats, g_labels, g_files, q_feats, q_labels, q_files, args):
    class_map = load_csv_file(args.label_file, to_int=True, frist_name=True)
    save_root, text_size = args.save_root, args.text_size
    visualizer = VisualizeResults(save_root, "search", text_size=text_size, class_map=class_map, only_error=True)

    faiss_index = create_index(g_feats, use_gpu=args.use_gpu, param=args.param, measure=args.measure)
    scores, index = faiss_index.search(q_feats, args.topk)
    p_labels, p_files = g_labels[index], g_files[index]
    visualizer.do_visualize(q_labels, q_files, p_labels, p_files, scores)


if __name__ == "__main__":
    args = parse_args()
    args.param = "IVF629,Flat"
    args.measure = faiss.METRIC_INNER_PRODUCT
    # 加载npz文件
    g_feats, g_labels, g_files = load_data(args.gallerys)
    q_feats, q_labels, q_files = load_data(args.querys)

    if args.debug:
        mask_files = np.load(args.mask_path)
        masks = np.isin(q_files, mask_files)
        keeps = np.array(np.arange(q_files.shape[0]))[~masks]
        q_feats, q_labels, q_files = q_feats[keeps], q_labels[keeps], q_files[keeps]

    with open(args.cats_file, "r") as f:
        class_list = np.array([int(line.strip("\n")) for line in f.readlines()])
    q_labels = class_list[q_labels]
    need_cats = np.array([999921340, 10000002454, 999920652])
    choices = np.isin(q_labels, need_cats)
    keeps = np.where(choices == True)[0]
    q_feats, q_labels, q_files = q_feats[keeps], q_labels[keeps], q_files[keeps]

    faiss.normalize_L2(g_feats)
    faiss.normalize_L2(q_feats)
    vis_error(g_feats, g_labels, g_files, q_feats, q_labels, q_files, args)
