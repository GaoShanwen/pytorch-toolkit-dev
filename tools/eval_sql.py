######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.04.12
# filenaem: eval_sql.py
# function: eval the accuracy of sql data date=xxxx-xx-xx, brand_id=xxxx.
######################################################
import argparse
import numpy as np

from local_lib.utils.feat_tools import run_compute, compute_acc_by_cat
from local_lib.utils.file_tools import save_dict2csv, create_sql_server, read_sql_data


def parse_args():
    parser = argparse.ArgumentParser(description="Create SQL Arguments")
    parser.add_argument("--instance-idx", type=int, default=0)
    parser.add_argument("--custom-keys", type=str, 
        default="sBarcode,sImgUrl,sTradeFlowNo,dCreateDate,src")
    parser.add_argument("--brand-id", type=int, default=1386)
    parser.add_argument("--set-date", type=str, nargs='*', default=None)
    parser.add_argument("--set-cats", type=str, nargs='*', default=None)
    parser.add_argument("--label-names-path", type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    sql_server = create_sql_server(args.brand_id, args.custom_keys)
    product_ids, gts, urls, dates, preds = read_sql_data(sql_server, args.set_date, args.set_cats)

    assert gts.shape[0] >= 1, "the sql data number must be greater than 0!"
    compare_preds, compare_gts = np.full((gts.shape[0], 5), np.nan), np.zeros(gts.shape[0]).astype(int)
    product_idx, this_prodict_code = 0, None
    for idx, (product_id, gt, url, date, pred) in enumerate(zip(product_ids, gts, urls, dates, preds)):
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
    for key, value in acc_map.items():
        value.pop("gallery_num")
    save_dict2csv(acc_map, f"test_{args.brand_id}_{args.set_date}.csv")
