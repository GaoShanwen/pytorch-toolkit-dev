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
    parser.add_argument(
        "--pass-remove", action="store_true", default=False, help="pass remove redundancy flag(False: run remove)"
    )
    parser.add_argument("--pass-cats", type=str, default="dataset/removeredundancy/pass_cats.txt")
    parser.add_argument("--mask-path", type=str, default="blacklist-val.npy")
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
        class_list = [line.strip("\n") for line in f.readlines()]
    label_index = load_csv_file(args.label_file)
    cats = list(set(g_label))
    label_map = {i: label_index[cat] for i, cat in enumerate(class_list) if i in cats}
    search_and_vis(g_feats, g_label, g_files, q_feats, q_label, q_files, args, label_map)
