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
from local_lib.utils.visualize import VisualizeResults
from local_lib.utils.visualize.results import save_imgs
from local_lib.utils.file_tools import load_data, save_dict2csv, save_keeps2mysql, load_names
# from local_lib.utils.visualize import save_imgs load_csv_file, 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gallerys", type=str)
    parser.add_argument("-q", "--querys", type=str)
    parser.add_argument("--save-root", type=str, default="output/vis/errors")
    parser.add_argument("--label-file", type=str, default="dataset/zero_dataset/label_names.csv")
    parser.add_argument("--set-files", type=str, default="dataset/removeredundancy/pass_cats.txt")
    parser.add_argument("--remove-mode", type=str, default=None, help="noise or similarity(default:None)")
    parser.add_argument("--vis-way", type=str, default=None, help="visualize way(eq:classify or search)")
    parser.add_argument("--filter-mode", type=str, default=None, help="label or file_path(default:None)")
    parser.add_argument("--mask-path", type=str, default="", help="blacklist-train.npz")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--run-test", action="store_true", default=False)
    parser.add_argument("--save-detail", action="store_true", default=False)
    parser.add_argument("--pass-mapping", action="store_true", default=False)
    parser.add_argument("--save-sql", action="store_true", default=False)
    parser.add_argument("--do-keep", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--similarity-th", type=float, default=None)
    parser.add_argument("--final-th", type=float, default=None)
    parser.add_argument("--trick-id", type=int, default=None)
    parser.add_argument("--update-times", type=int, default=0)
    parser.add_argument("--idx-column", type=int, default=0)
    parser.add_argument("--name-column", type=int, default=-1)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def eval_server(g_feats, g_label, q_feats, q_label, args, acc_file_name="eval_res.csv"):
    index = feat_tools.create_index(g_feats, use_gpu=args.use_gpu, param=args.param, measure=args.measure)
    D, I = index.search(q_feats, args.topk)
    p_label, p_scores = feat_tools.get_predict_label(D, I, g_label, threshold=args.similarity_th, trick_id=args.trick_id)
    if acc_file_name:
        label_map = load_names(args.label_file, idx_column=args.idx_column, name_column=args.name_column)
        acc_map = feat_tools.compute_acc_by_cat(p_label, q_label, p_scores, label_map, threshold=args.final_th)
        for key, value in acc_map.items():
            g_num = np.sum(np.isin(g_label, [int(key)])) if key != "all_data" else g_label.shape[0]
            value.update({"gallery_num": g_num})
        save_dict2csv(acc_map, acc_file_name)
    
    feat_tools.run_compute(p_label, q_label, p_scores, do_output=True, th=args.final_th)


def main(g_feats, g_label, g_files, q_feats, q_label, q_files, args):
    eval_server(g_feats, g_label, q_feats, q_label, args, "eval_res.csv")
    if args.vis_way is not None:
        save_root, text_size = args.save_root, 48
        index = feat_tools.create_index(g_feats, use_gpu=args.use_gpu, param=args.param, measure=args.measure)
        D, I = index.search(q_feats, args.topk)
        label_maps = load_names(args.label_file, idx_column=args.idx_column, name_column=args.name_column)
        visualizer = VisualizeResults(save_root, args.vis_way, text_size, label_maps, ["show_gt"]) #"only_error", 
        if args.vis_way == "search":
            visualizer.do_visualize(q_label, q_files, g_label[I], p_files=g_files[I], scores=D)
            return
        
        p_label, p_scores = feat_tools.get_predict_label(D, I, g_label, 5, args.similarity_th, args.trick_id)
        visualizer.do_visualize(q_label, q_files, p_label, p_files=None, scores=p_scores)
    if args.mask_path:
        data = np.load(args.mask_path)
        masks = np.isin(g_files, data["files"])
        keeps = np.array(np.arange(g_files.shape[0]))[~masks]
        print(f"original data: {g_label.shape[0]}, after remove blacklist data: {keeps.shape[0]}")
        g_feats, g_label, g_files = g_feats[keeps], g_label[keeps], g_files[keeps]
    if args.remove_mode is None:
        if args.save_sql:
            save_keeps2mysql(g_feats, g_label, g_files, update_times=args.update_times)
        return
    keeps = feat_tools.run_choose(g_feats, g_label, args)
    # if args.remove_mode == "choose_noise":
    #     if args.save_detail:
    #         label_index = load_names(args.label_file)
    #         label_map = {int(cat): name.split("/")[0] for cat, name in label_index.items()}
    #         save_imgs(new_g_files, new_g_label, label_map, args.save_root)
    #     all_indics = np.arange(g_label.shape[0])
    #     keeps = np.setdiff1d(all_indics, keeps)

    print(f"original data: {g_label.shape[0]}, after remove {args.remove_mode} data: {keeps.shape[0]}")
    new_g_feats, new_g_label, new_g_files = g_feats[keeps], g_label[keeps], g_files[keeps]
    save_path = f"output/feats/regnety_040-train-{args.remove_mode}-{args.similarity_th}.npz"
    np.savez(save_path, feats=new_g_feats, gts=new_g_label, fpaths=new_g_files)
    if args.save_sql:
        save_keeps2mysql(new_g_feats, new_g_label, new_g_files, update_times=args.update_times)
    csv_name = f"eval_res-{args.remove_mode}-{args.similarity_th}.csv" if args.save_detail else ""
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
        label_index = load_names(args.label_file, idx_column=args.idx_column, name_column=args.name_column)
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
    # args.param = "IVF629,Flat"
    args.param = "Flat"
    args.measure = faiss.METRIC_INNER_PRODUCT

    # 加载npz文件
    g_feats, g_label, g_files = load_data(args.gallerys)
    g_files = np.array([path.replace("/exp-data", "") for path in g_files])
    g_files = np.array([file.replace("./dataset/function_test/test_subo/train", "/data/AI-scales/images/1173/backflow") for file in g_files])
    if args.querys:
        q_feats, q_label, q_files = load_data(args.querys)
        q_files = np.array([path.replace("/exp-data", "") for path in q_files])
    q_files = np.array([file.replace("./dataset/function_test/test_subo/val", "/data/AI-scales/images/1173/backflow") for file in q_files])

    if args.filter_mode is not None:
        assert args.filter_mode in ["label", "file_path"], "filter_model must be label or file_path!"
        if args.filter_mode == "label":
            with open(args.set_files, "r") as f:
                g_objects = [int(line.strip("\n")) for line in f.readlines()]
            q_objects = g_objects
        else:
            data = np.load(args.set_files)
            g_objects, q_objects = data["g_files"], data["q_files"]
        print(f"set {args.filter_mode} g_number={len(g_objects)}, q_number={len(q_objects)}")
        org_object = g_label if args.filter_mode == "label" else g_files
        choice = np.isin(org_object, g_objects)
        keeps = choice if args.do_keep else ~choice
        g_feats, g_label, g_files = g_feats[keeps], g_label[keeps], g_files[keeps]

        org_object = q_label if args.filter_mode == "label" else q_files
        choice = np.isin(org_object, q_objects)
        keeps = choice if args.do_keep else ~choice
        q_feats, q_label, q_files = q_feats[keeps], q_label[keeps], q_files[keeps]
    cats = list(set(g_label))
    print(f"Loaded cats number={len(cats)}, gallery number={g_label.shape[0]}")

    faiss.normalize_L2(g_feats)
    if args.querys:
        faiss.normalize_L2(q_feats)
    else:
        q_feats, q_label, q_files = None, None, None

    # keeps = np.isin(q_label, [40038,])
    # q_feats, q_label, q_files = q_feats[keeps], q_label[keeps], q_files[keeps]
    # import pdb; pdb.set_trace()
    print(f"Loaded cats number={len(cats)}, query number={q_label.shape[0]}")
    function_name = "run_test" if args.run_test else "main"
    eval(function_name)(g_feats, g_label, g_files, q_feats, q_label, q_files, args)
