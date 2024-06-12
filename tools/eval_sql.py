######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.04.12
# filenaem: eval_sql.py
# function: eval the accuracy of sql data date=xxxx-xx-xx, brand_id=xxxx.
######################################################
import argparse
import numpy as np

from local_lib.utils.file_tools import MySQLHelper
from local_lib.utils.feat_tools import run_compute, compute_acc_by_cat
from local_lib.utils.file_tools import save_dict2csv, load_names


def parse_args():
    parser = argparse.ArgumentParser(description="Create SQL Arguments")
    parser.add_argument("--set-date", type=str, default=None)
    parser.add_argument("--length", type=int, default=38)
    parser.add_argument("--instance-idx", type=int, default=0)
    parser.add_argument("--groundture-column", type=int, default=1)
    parser.add_argument("--product-column", type=int, default=11)
    parser.add_argument("--date-column", type=int, default=16)
    parser.add_argument("--predict-column", type=int, default=20)
    parser.add_argument("--idx-column", type=int, default=0)
    parser.add_argument("--name-column", type=int, default=-1)
    parser.add_argument("--brand-id", type=int, default=1345)
    parser.add_argument("--url-column", type=int, default=6)
    return parser.parse_args()


def create_sql_server(brand_id):
    assert brand_id is not None, f"please set a value for brand_id!"
    private_hostmap= {479: 1, 1000: 4, 684: 0}
    host_base = brand_id // 100 if brand_id not in private_hostmap else private_hostmap[brand_id]
    return MySQLHelper(
        host=f"balance-open{host_base:02d}.mysql.rds.aliyuncs.com",
        port=3350,
        user="rueread",
        password="read1107!@#$",
        database=f"balance{brand_id}",
        table_name="tgoodscollectpic",
    )


if __name__ == '__main__':
    args = parse_args()
    sql_server = create_sql_server(args.brand_id)
    read_res = sql_server.read_table()
    set_length, gt_column, pred_column = args.length, args.groundture_column, args.predict_column
    product_id_column = args.product_column
    read_res = [res for res in read_res if res[args.date_column]==args.set_date]
    # read_res = [res for res in read_res if res[args.date_column].startswith("2024-05-2")]
    assert len(read_res) >= 1, "the sql data number must be greater than 0!"
    predicts, gts = np.full((len(read_res), 5), np.nan), np.zeros(len(read_res)).astype(int)
    product_idx, this_prodict_code = 0, None
    for idx, instance in enumerate(read_res):
        if len(instance) != set_length:
            print(f"line={instance[args.instance_idx]} is error, please check ! {instance}")
            continue
        if not len(instance[pred_column]):
            continue
        pred_res = eval(instance[pred_column])
        if not len(pred_res):
            continue
        predicts[product_idx, :len(pred_res)] = np.array([pred["label"] for pred in pred_res])
        gts[product_idx] = instance[gt_column]
        if this_prodict_code!= instance[product_id_column]:
            this_prodict_code = instance[product_id_column]
            product_idx += 1
    masks = np.zeros(gts.shape[0]).astype(bool) #np.isin(gts, [0])
    masks[product_idx+1:] = True
    predicts, gts = predicts[~masks], gts[~masks]
    run_compute(predicts, gts, do_output=True)
    # idx_column, name_column = args.idx_column, args.name_column
    label_map = {int(product_id): name for product_id, name in sql_server.read_names()}
    acc_map = compute_acc_by_cat(predicts, gts, label_map=label_map)
    for key, value in acc_map.items():
        value.pop("gallery_num")
    save_dict2csv(acc_map, f"test_{args.brand_id}_{args.set_date}.csv")
