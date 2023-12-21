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
from local_lib.utils.visualize import save_imgs
from local_lib.utils.file_tools import save_keeps2mysql, load_csv_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gallerys", type=str)
    parser.add_argument("-q", "--querys", type=str)
    parser.add_argument("--save-root", type=str, default="output/vis/noises")
    parser.add_argument("--label-file", type=str, default="dataset/zero_dataset/label_names.csv")
    parser.add_argument("--cats-file", type=str, default="dataset/removeredundancy/629_cats.txt")
    parser.add_argument("--pass-cats", type=str, default="dataset/removeredundancy/pass_cats.txt")
    parser.add_argument("--mask-path", type=str, default="blacklist-val.npy")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--use-knn", action="store_true", default=False)
    parser.add_argument("--run-test", action="store_true", default=False)
    parser.add_argument("--weighted", action="store_true", default=False)
    parser.add_argument("--save-detail", action="store_true", default=False)
    parser.add_argument("--pass-mapping", action="store_true", default=False)
    parser.add_argument("--remove-mode", type=str, default="none", help="remove mode(eq:noise, or similarity)!")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--save-sql", action="store_true", default=False)
    parser.add_argument("--num-classes", type=int, default=629)
    parser.add_argument("--update-times", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def run_eval(g_feats, g_label, q_feats, q_label, args, acc_file_name="eval_res.csv"):
    index = feat_tools.create_index(g_feats, use_gpu=args.use_gpu, param=args.param, measure=args.measure)
    D, I = index.search(q_feats, args.topk)
    p_label = feat_tools.get_predict_label(D, I, g_label, use_knn=args.use_knn, weighted=args.weighted)
    if acc_file_name:
        label_index = load_csv_file(args.label_file)
        label_map = {int(cat): name.split("/")[0] for cat, name in label_index.items()}
        acc_map = feat_tools.compute_acc_by_cat(p_label, q_label, label_map)
        feat_tools.print_acc_map(acc_map, acc_file_name)
    feat_tools.run_compute(p_label, q_label, do_output=False)


def main(g_feats, g_label, g_files, q_feats, q_label, q_files, args):
    run_eval(g_feats, g_label, q_feats, q_label, args, "eval_res.csv")

    if args.remove_mode == "none":
        if args.save_sql:
            save_keeps2mysql(g_feats, g_label, g_files, update_times=args.update_times)
        return
    keeps = feat_tools.run_choose(g_feats, g_label, args)
    if args.remove_mode == "choose_noise":
        if args.save_detail:
            label_index = load_csv_file(args.label_file)
            label_map = {int(cat): name.split("/")[0] for cat, name in label_index.items()}
            save_imgs(new_g_files, new_g_label, label_map, args.save_root)
        all_indics = np.arange(g_label.shape[0])
        keeps = np.setdiff1d(all_indics, keeps)

    print(f"original data: {g_label.shape[0]}, after remove {args.remove_mode} data: {keeps.shape[0]}")
    new_g_feats, new_g_label, new_g_files = g_feats[keeps], g_label[keeps], g_files[keeps]
    save_path = f"output/feats/regnety_040-train-{args.remove_mode}-{args.threshold}.npz"
    np.savez(save_path, feats=new_g_feats, gts=new_g_label, fpaths=new_g_files)
    if args.save_sql:
        save_keeps2mysql(new_g_feats, new_g_label, new_g_files, update_times=args.update_times)
    # feat_tools.save_keeps_file(new_g_label, new_g_files, f"train-th{args.threshold}.txt")
    csv_name = f"eval_res-{args.remove_mode}-{args.threshold}.csv" if args.save_detail else ""
    run_eval(new_g_feats, new_g_label, q_feats, q_label, args, csv_name)


def run_test(g_feats, g_label, g_files, q_feats, q_label, q_files, args):
    run_eval(g_feats, g_label, q_feats, q_label, args, acc_file_name="")
    mask_files = np.load(args.mask_path)
    masks = np.isin(q_files, mask_files)
    keeps = np.array(np.arange(q_files.shape[0]))[~masks]
    print(f"original data: {q_label.shape[0]}, after remove blacklist data: {keeps.shape[0]}")
    q_feats, q_label, q_files = q_feats[keeps], q_label[keeps], q_files[keeps]
    # masks = np.isin(g_files, mask_files)
    # keeps = np.array(np.arange(g_files.shape[0]))[~masks]
    # print(f"original data: {g_label.shape[0]}, after remove blacklist data: {keeps.shape[0]}")
    # g_feats, g_label, g_files = g_feats[keeps], g_label[keeps], g_files[keeps]
    run_eval(g_feats, g_label, q_feats, q_label, args, acc_file_name="eval_res.csv")
    # #################### static topk ####################
    if args.remove_mode == "static":
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
        feat_tools.print_acc_map(new_static, "static.csv")
        return
    #################### test knn ####################
    if args.use_knn:
        index = feat_tools.create_index(g_feats, use_gpu=args.use_gpu, param=args.param, measure=args.measure)
        D, I = index.search(q_feats, args.topk)
        p_label = feat_tools.get_predict_label(D, I, g_label, q_label, use_knn=False)
        original_errors = np.where(p_label[:, 0] != q_label)[0]
        p_label = feat_tools.get_predict_label(D, I, g_label, q_label, use_knn=True)
        knn_errors = np.where(p_label[:, 0] != q_label)[0]

        e_only_in_knn = np.setdiff1d(knn_errors, original_errors)
        print(
            f"new-errors knn-k={args.topk}: {e_only_in_knn.shape[0]} / {q_label.shape[0]} | "
            f"({original_errors.shape[0]} -> {knn_errors.shape[0]})"
        )
        new_q_labels, new_q_files = q_label[e_only_in_knn], q_files[e_only_in_knn]
        feat_tools.save_keeps_file(new_q_labels, new_q_files, f"new_errors-knn.txt")
        return


def add_blacklist(feats, labels, files):
    feat_path = "output/feats/blacklist-train.npz"
    exp_feats, exp_label, exp_files = feat_tools.load_data(feat_path)
    exp_f = []
    for label, file_path in zip(exp_label, exp_files):
        exp_f.append(
            file_path.replace(
                f"dataset/blacklist2/{label:08d}", f"/data/AI-scales/images/0/backflow/{label+110110110001}"
            )
        )
    print(f"Loaded blacklist: {exp_label.shape[0]}")
    exp_label += 110110110001
    feats = np.concatenate((exp_feats, feats), axis=0) if feats is not None else feats
    labels = np.concatenate((exp_label, labels)) if labels is not None else labels
    files = np.concatenate((exp_files, files)) if files is not None else files
    return feats, labels, files


if __name__ == "__main__":
    args = parse_args()
    # args.param = "Flat"
    args.param = "IVF629,Flat"
    args.measure = faiss.METRIC_INNER_PRODUCT

    # 加载npz文件
    g_feats, g_label, g_files = feat_tools.load_data(args.gallerys)
    q_feats, q_label, q_files = feat_tools.load_data(args.querys)

    with open(args.cats_file, "r") as f:
        class_list = np.array([int(line.strip("\n")) for line in f.readlines()])
        g_label = g_label if args.pass_mapping else class_list[g_label]
        q_label = class_list[q_label]

    if args.debug:
        choose_cats = list(set(g_label))  # [:50]
        print(f"origain cats number={len(choose_cats)}")
        with open("dataset/removeredundancy/629_cats.txt", "r") as f:
            choose_cats = [int(line.strip("\n")) for line in f.readlines()]
        print(f"final cats number={len(choose_cats)}")
        cat_idx = np.where(g_label[:, np.newaxis] == choose_cats)[0]
        g_feats, g_label, g_files = g_feats[cat_idx], g_label[cat_idx], g_files[cat_idx]

        cat_idx = np.where(q_label[:, np.newaxis] == choose_cats)[0]
        q_feats, q_label, q_files = q_feats[cat_idx], q_label[cat_idx], q_files[cat_idx]

    g_feats, g_label, g_files = add_blacklist(g_feats, g_label, g_files)
    faiss.normalize_L2(g_feats)
    faiss.normalize_L2(q_feats)
    function_name = "run_test" if args.run_test else "main"
    eval(function_name)(g_feats, g_label, g_files, q_feats, q_label, q_files, args)
