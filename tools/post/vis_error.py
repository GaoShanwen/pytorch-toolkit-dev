######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.16
# filenaem: vis_error.py
# function: visualize the error picture
######################################################
import argparse

import faiss
import numpy as np

from local_lib.utils.file_tools import load_csv_file, load_data
from local_lib.utils.visualize import search_and_vis


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
    parser.add_argument("--num-classes", type=int, default=629)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--text-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=9)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # args.param = f"IVF{args.num_classes},Flat"
    args.param = "IVF150,Flat"
    args.measure = faiss.METRIC_INNER_PRODUCT
    # 加载npz文件
    g_feats, g_label, g_files = load_data(args.gallerys)
    q_feats, q_label, q_files = load_data(args.querys)
    faiss.normalize_L2(g_feats)
    faiss.normalize_L2(q_feats)

    if args.debug:
        mask_files = np.load(args.mask_path)
        masks = np.isin(q_files, mask_files)
        keeps = np.array(np.arange(q_files.shape[0]))[~masks]
        q_feats, q_label, q_files = q_feats[keeps], q_label[keeps], q_files[keeps]

    with open(args.cats_file, "r") as f:
        class_list = np.array([int(line.strip("\n")) for line in f.readlines()])
    q_label = class_list[q_label]
    need_cats = np.array(
        [
            10000003015,
            9999151125,
            999925408,
            9999150874,
            9999151657,
            999925207,
            999920270,
            9999151662,
            10000000302,
            9999150297,
            9999150301,
            999925228,
            999925212,
        ]
    )
    keep = np.isin(q_label, need_cats)
    keeps = np.array(np.arange(q_files.shape[0]))[keep]
    q_feats, q_label, q_files = q_feats[keeps], q_label[keeps], q_files[keeps]

    label_index = load_csv_file(args.label_file)
    label_map = {int(cat): name for cat, name in label_index.items()}
    search_and_vis(g_feats, g_label, g_files, q_feats, q_label, q_files, args, label_map)

    # if args.visualize:
    #     class_map = load_csv_file(args.label_file)
    #     class_to_idx = dataset.reader.class_to_idx
    #     idx_to_names = {idx: class_map[p_ids] for p_ids, idx in class_to_idx.items()}
    #     visualizer = VisualizeResults(args.results_file, class_map=idx_to_names)

    #     visualizer.do_visualize()
