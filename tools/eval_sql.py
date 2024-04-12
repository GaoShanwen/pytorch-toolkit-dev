import argparse
import numpy as np

from local_lib.utils.file_tools import MySQLHelper
from local_lib.utils.feat_tools import run_compute, compute_acc_by_cat
from local_lib.utils.file_tools import save_dict2csv, load_names


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-date", type=str, default=None)
    parser.add_argument("--length", type=int, default=37)
    parser.add_argument("--instance-idx", type=int, default=0)
    parser.add_argument("--groundture-column", type=int, default=1)
    parser.add_argument("--product-column", type=int, default=10)
    parser.add_argument("--date-column", type=int, default=15)
    parser.add_argument("--predict-column", type=int, default=19)
    parser.add_argument("--idx-column", type=int, default=0)
    parser.add_argument("--name-column", type=int, default=-1)
    parser.add_argument("--label-file", type=str, default=None)
    parser.add_argument("--brand-id", type=int, default=1345)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.set_date is not None, f"please set a value for set_date!"
    sql_server = MySQLHelper(
        host=f"balance-open{args.brand_id//100}.mysql.rds.aliyuncs.com",
        port=3350,
        user="rueread",
        password="read1107!@#$",
        database=f"balance{args.brand_id}",
        table_name="tgoodscollectpic",
    )
    read_res = sql_server.read_table()
    set_length, gt_column, pred_column = args.length, args.groundture_column, args.predict_column
    product_column = args.product_column
    read_res = [res for res in read_res if res[args.date_column]==args.set_date]
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
        if this_prodict_code!= instance[product_column]:
            this_prodict_code = instance[product_column]
            product_idx += 1
        predicts[product_idx, :len(pred_res)] = np.array([pred["label"] for pred in pred_res])
        gts[product_idx] = instance[gt_column]
    masks = np.zeros(gts.shape[0]).astype(bool) #np.isin(gts, [0])
    masks[product_idx+1:] = True
    predicts, gts = predicts[~masks], gts[~masks]
    run_compute(predicts, gts, do_output=True)
    label_file, idx_column, name_column = args.label_file, args.idx_column, args.name_column
    if label_file is not None:
        label_map = load_names(label_file, idx_column=idx_column, name_column=name_column)
    acc_map = compute_acc_by_cat(predicts, gts, label_map=label_map)
    # for key, value in acc_map.items():
    #     g_num = np.sum(np.isin(gts, [int(key)])) if key != "all_data" else gts.shape[0]
    #     value.update({"gallery_num": g_num})
    for key, value in acc_map.items():
        value.pop("gallery_num")
    save_dict2csv(acc_map, "test_1345.csv")
    # import pdb; pdb.set_trace()
