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

from local_lib.utils import feat_tools
from local_lib.utils.visualize.results import save_imgs
from local_lib.utils.file_tools import load_csv_file, load_data, save_dict2csv, save_keeps2mysql
# from local_lib.utils.visualize import save_imgs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gallerys", type=str)
    parser.add_argument("-q", "--querys", type=str)
    parser.add_argument("--save-root", type=str, default="output/vis/noises")
    parser.add_argument("--label-file", type=str, default="dataset/zero_dataset/label_names.csv")
    parser.add_argument("--pass-cats", type=str, default="dataset/removeredundancy/pass_cats.txt")
    parser.add_argument("--remove-mode", type=str, default="none", help="remove mode(eq:noise, or similarity)!")
    parser.add_argument("--mask-path", type=str, default="", help="blacklist-train.npz")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--use-knn", action="store_true", default=False)
    parser.add_argument("--run-test", action="store_true", default=False)
    parser.add_argument("--weighted", action="store_true", default=False)
    parser.add_argument("--save-detail", action="store_true", default=False)
    parser.add_argument("--pass-mapping", action="store_true", default=False)
    parser.add_argument("--save-sql", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--update-times", type=int, default=0)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def eval_server(g_feats, g_label, q_feats, q_label, args, acc_file_name="eval_res.csv"):
    index = feat_tools.create_index(g_feats, use_gpu=args.use_gpu, param=args.param, measure=args.measure)
    D, I = index.search(q_feats, args.topk)
    p_label = feat_tools.get_predict_label(D, I, g_label, use_knn=args.use_knn, weighted=args.weighted)
    if acc_file_name:
        label_index = load_csv_file(args.label_file)
        label_map = {int(cat): name.split("/")[0] for cat, name in label_index.items()}
        acc_map = feat_tools.compute_acc_by_cat(p_label, q_label, label_map)
        save_dict2csv(acc_map, acc_file_name)
    feat_tools.run_compute(p_label, q_label, do_output=False)


def main(g_feats, g_label, g_files, q_feats, q_label, q_files, args):
    eval_server(g_feats, g_label, q_feats, q_label, args, "eval_res.csv")
    if args.mask_path:
        data = np.load(args.mask_path)
        # import pdb; pdb.set_trace()
        masks = np.isin(g_files, data["files"])
        keeps = np.array(np.arange(g_files.shape[0]))[~masks]
        print(f"original data: {g_label.shape[0]}, after remove blacklist data: {keeps.shape[0]}")
        g_feats, g_label, g_files = g_feats[keeps], g_label[keeps], g_files[keeps]
    if args.remove_mode == "none":
        if args.save_sql:
            save_keeps2mysql(g_feats, g_label, g_files, update_times=args.update_times)
        return
    keeps = feat_tools.run_choose(g_feats, g_label, args)
    # if args.remove_mode == "choose_noise":
    #     if args.save_detail:
    #         label_index = load_csv_file(args.label_file)
    #         label_map = {int(cat): name.split("/")[0] for cat, name in label_index.items()}
    #         save_imgs(new_g_files, new_g_label, label_map, args.save_root)
    #     all_indics = np.arange(g_label.shape[0])
    #     keeps = np.setdiff1d(all_indics, keeps)

    print(f"original data: {g_label.shape[0]}, after remove {args.remove_mode} data: {keeps.shape[0]}")
    new_g_feats, new_g_label, new_g_files = g_feats[keeps], g_label[keeps], g_files[keeps]
    save_path = f"output/feats/regnety_040-train-{args.remove_mode}-{args.threshold}.npz"
    np.savez(save_path, feats=new_g_feats, gts=new_g_label, fpaths=new_g_files)
    if args.save_sql:
        save_keeps2mysql(new_g_feats, new_g_label, new_g_files, update_times=args.update_times)
    csv_name = f"eval_res-{args.remove_mode}-{args.threshold}.csv" if args.save_detail else ""
    eval_server(new_g_feats, new_g_label, q_feats, q_label, args, csv_name)


def return_samilirity_cats(static_v, th):
    return {
        static_v[f"top{idx}_name"]: static_v[f"top{idx}_ratio"]
        for idx in range(2, 6)
        if static_v[f"top{idx}_ratio"] >= th
    }


def print_static(static_res, th=0.01):
    cats = list(static_res.keys())
    masks = np.logical_not(np.ones(len(static_res)))

    for i, (k, v) in enumerate(static_res.items()):
        if masks[i]:
            continue
        check_objs = list(return_samilirity_cats(v, th).items())
        print(f"{k}: {check_objs}")
        masks[i] = True
        while len(check_objs):
            obj = check_objs.pop(0)[0]
            idx = cats.index(obj)
            if masks[idx]:
                continue
            searched = list(return_samilirity_cats(static_res[obj], th).items())
            print(f"{obj}: {searched}")
            check_objs += searched
            masks[idx] = True
        print()
        # import pdb; pdb.set_trace()


def run_test(g_feats, g_label, g_files, q_feats, q_label, q_files, args):
    import pdb; pdb.set_trace()
    keeps = feat_tools.run_choose(g_feats, g_label, args)
    new_g_files, new_g_label = g_files[keeps], g_label[keeps]
    save_imgs(new_g_files, new_g_label, args.save_root)
    # if args.remove_mode == "choose_noise":
    #     if args.save_detail:
    #         label_index = load_csv_file(args.label_file)
    #         label_map = {int(cat): name.split("/")[0] for cat, name in label_index.items()}
    #         save_imgs(new_g_files, new_g_label, label_map, args.save_root)
    #     all_indics = np.arange(g_label.shape[0])
    #     keeps = np.setdiff1d(all_indics, keeps)

    print(f"original data: {g_label.shape[0]}, after remove {args.remove_mode} data: {keeps.shape[0]}")
    # #################### static topk ####################
    if args.remove_mode == "choose_with_static":
        static = feat_tools.run_choose(g_feats, g_label, args)
        label_index = load_csv_file(args.label_file)
        label_map = {int(cat): name.split("/")[0] for cat, name in label_index.items()}
        new_static = {}
        for cat, value in static.items():
            new_value = {}
            for k, v in value.items():
                value_name = label_map[v] if k.endswith("name") else v
                new_value.update({k: value_name})
            new_static.update({label_map[cat]: new_value})
        print_static(new_static, 0.01)
        save_dict2csv(new_static, "static.csv")
        return
    eval_server(g_feats, g_label, q_feats, q_label, args, acc_file_name="")


def expend_feats(feats, labels, files, feat_path):
    exp_feats, exp_label, exp_files = load_data(feat_path)
    print(f"Loaded blacklist: {exp_label.shape[0]}")
    feats = np.concatenate((exp_feats, feats), axis=0) if feats is not None else feats
    labels = np.concatenate((exp_label, labels)) if labels is not None else labels
    files = np.concatenate((exp_files, files)) if files is not None else files
    return feats, labels, files


if __name__ == "__main__":
    args = parse_args()
    args.param = "IVF629,Flat"
    args.measure = faiss.METRIC_INNER_PRODUCT

    # 加载npz文件
    g_feats, g_label, g_files = load_data(args.gallerys)
    if args.querys:
        q_feats, q_label, q_files = load_data(args.querys)

    # with open(args.cats_file, "r") as f:
    #     class_list = np.array([int(line.strip("\n")) for line in f.readlines()])
    #     g_label = g_label if args.pass_mapping else class_list[g_label]
    #     q_label = class_list[q_label]

    # if not args.pass_mapping:
    #     feat_paths = ["output/feats/add_2c-train.npz", "output/feats/blacklist-train.npz"]
    #     for feat_path in feat_paths:
    #         g_feats, g_label, g_files = expend_feats(g_feats, g_label, g_files, feat_path)
    if args.debug:
        with open(args.pass_cats, "r") as f:
            choose_cats = [int(line.strip("\n")) for line in f.readlines()]
        print(f"remove cats number={len(choose_cats)}")
        masks = np.isin(g_label, choose_cats)
        g_feats, g_label, g_files = g_feats[~masks], g_label[~masks], g_files[~masks]

        masks = np.isin(q_label, choose_cats)
        q_feats, q_label, q_files = q_feats[~masks], q_label[~masks], q_files[~masks]
    cats = list(set(g_label))
    print(f"Loaded cats number={len(cats)}, img number={g_label.shape[0]}")

    faiss.normalize_L2(g_feats)
    if args.querys:
        faiss.normalize_L2(q_feats)
    else:
        q_feats, q_label, q_files = None, None, None
    function_name = "run_test" if args.run_test else "main"
    eval(function_name)(g_feats, g_label, g_files, q_feats, q_label, q_files, args)
