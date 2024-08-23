######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2023.11.09
# filenaem: eval_feats.py
# function: eval the accuracy of features in .npz-file.
######################################################
import argparse
import logging
import faiss
import numpy as np
from colorama import Fore, Style

from local_lib.utils.feat_tools import create_index, get_predict_label, compute_acc_by_cat, run_compute, run_choose, eval_server
from local_lib.utils.file_tools import load_data, save_dict2csv, save_keeps2mysql, load_names, create_sql_server, read_sql_data
from local_lib.utils.visualize import VisualizeResults


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(f"[{Fore.MAGENTA} evaluate {Style.RESET_ALL}]")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate features or SQL queries")
    parser.add_argument("--brand-id", type=int, default=None, help="brand id")
    parser.add_argument("--eval-sql", action="store_true", default=False)
    
    parser.add_argument("-g", "--gallerys", type=str, help="gallery features file path")
    parser.add_argument("-q", "--querys", type=str, help="query features file path")
    parser.add_argument("--save-root", type=str, default="output/vis/errors")
    parser.add_argument("--label-file", type=str, default="dataset/zero_dataset/label_names.csv")
    parser.add_argument("--set-files", type=str, default="dataset/removeredundancy/pass_cats.txt")
    parser.add_argument("--remove-mode", type=str, default=None, help="noise or similarity(default:None)")
    parser.add_argument("--vis-way", type=str, default=None, help="visualize way(eq:classify or search)")
    parser.add_argument("--filter-mode", type=str, default=None, help="label or file_path(default:None)")
    parser.add_argument("--mask-path", type=str, default="", help="blacklist-train.npz")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--save-detail", action="store_true", default=False)
    parser.add_argument("--pass-mapping", action="store_true", default=False)
    parser.add_argument("--choices-cats", type=int, nargs='*', default=None)
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

    parser.add_argument("--custom-keys", type=str, 
        default="sBarcode,sStoreId,sImgUrl,sTradeFlowNo,dCreateDate,src")
    parser.add_argument("--set-date", type=str, nargs='*', default=None)
    parser.add_argument("--set-cats", type=str, nargs='*', default=None)
    return parser.parse_args()


def get_label_names(args, sql_server):
    if sql_server is not None:
        return {int(product_id): name for product_id, name in sql_server.read_names()}
    return load_names(args.label_file, idx_column=args.idx_column, name_column=args.name_column)


def eval_feats(args, sql_server):
    # args.param = "IVF629,Flat"
    args.param = "Flat"
    args.measure = faiss.METRIC_INNER_PRODUCT

    # 加载npz文件
    g_feats, g_label, g_files = load_data(args.gallerys)
    if args.querys:
        q_feats, q_label, q_files = load_data(args.querys)

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
    _logger.info(f"Loaded cats number={len(cats)}, gallery number={g_label.shape[0]}")

    faiss.normalize_L2(g_feats)
    if args.querys:
        faiss.normalize_L2(q_feats)
    else:
        q_feats, q_label, q_files = None, None, None

    _logger.info(f"Loaded cats number={len(cats)}, query number={q_label.shape[0]}")
    # g_feats, q_feats = g_feats.round(decimals=3), q_feats.round(decimals=3)
    label_map = get_label_names(args, sql_server)

    eval_server(g_feats, g_label, q_feats, q_label, args, label_map, "eval_res.csv")
    if args.vis_way is not None:
        save_root, text_size = args.save_root, 48
        index = create_index(g_feats, use_gpu=args.use_gpu, param=args.param, measure=args.measure)
        keeps = np.isin(q_label, args.choices_cats) if args.choices_cats else np.ones(q_label.shape[0], dtype=bool)
        q_feats, q_label, q_files = q_feats[keeps], q_label[keeps], q_files[keeps]
        D, I = index.search(q_feats, args.topk)
        visualizer = VisualizeResults(save_root, args.vis_way, text_size, label_map, ["show_gt"]) #"only_error", 
        if args.vis_way == "search":
            visualizer.do_visualize(q_label, q_files, g_label[I], p_files=g_files[I], scores=D)
            return
        
        p_label, p_scores = get_predict_label(D, I, g_label, 5, args.similarity_th, args.trick_id)
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
    keeps = run_choose(g_feats, g_label, args)

    print(f"original data: {g_label.shape[0]}, after remove {args.remove_mode} data: {keeps.shape[0]}")
    new_g_feats, new_g_label, new_g_files = g_feats[keeps], g_label[keeps], g_files[keeps]
    save_path = f"output/feats/regnety_040-train-{args.remove_mode}-{args.similarity_th}.npz"
    np.savez(save_path, feats=new_g_feats, gts=new_g_label, fpaths=new_g_files)
    if args.save_sql:
        save_keeps2mysql(new_g_feats, new_g_label, new_g_files, update_times=args.update_times)
    csv_name = f"eval_res-{args.remove_mode}-{args.similarity_th}.csv" if args.save_detail else ""
    eval_server(new_g_feats, new_g_label, q_feats, q_label, args, label_map, csv_name)


def eval_sql(args, sql_server):
    gts, _, _, product_ids, _, preds = read_sql_data(sql_server, args.set_date, args.set_cats, args.set_stores)

    compare_preds, compare_gts = np.full((gts.shape[0], 5), np.nan), np.zeros(gts.shape[0]).astype(int)
    product_idx, this_prodict_code = 0, None
    for product_id, gt, pred in zip(product_ids, gts, preds):
        pred_res = eval(pred) if len(pred) else pred
        if not len(pred_res):
            continue
        compare_preds[product_idx, :len(pred_res)] = np.array([pred["label"] for pred in pred_res])
        compare_gts[product_idx] = gt
        if this_prodict_code != product_id:
            this_prodict_code = product_id
            product_idx += 1
    preds, gts = compare_preds[:product_idx], compare_gts[:product_idx]
    run_compute(preds, gts, do_output=True)

    label_map = {int(product_id): name for product_id, name in sql_server.read_names()}
    acc_map = compute_acc_by_cat(preds, gts, label_map=label_map)
    save_dict2csv(acc_map, f"test_{args.brand_id}_{args.set_date}.csv")


if __name__ == "__main__":
    args = parse_args()
    eval_obj = "sql" if args.eval_sql else "feats"
    sql_server = create_sql_server(args.brand_id, args.custom_keys) if args.brand_id else None
    eval(f"eval_{eval_obj}")(args, sql_server)
