######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.04.12
# filenaem: create_sql_data.py
# function: create csv file of sql data date=xxxx-xx-xx, brand_id=xxxx.
######################################################
import tqdm
import argparse
import numpy as np

from local_lib.utils.file_tools import MySQLHelper
from local_lib.utils.file_tools import save_dict2csv

from tools.eval_sql import parse_args, create_sql_server

def run_sql_with_save(sql_server, args):
    if args.label_names_path is not None:
        label_map = {product_id: name for product_id, name in sql_server.read_names()}
        save_dict2csv(label_map, args.label_names_path, index=['name'])
        return

    read_res = sql_server.read_table()
    assert len(read_res) >= 1, "the sql data number must be greater than 0!"
    assert len(read_res[0]) == args.length, "the sql data number must be equal with set length!"
    product_id_column, gt_column, url_column = args.product_column, args.groundture_column, args.url_column
    product_ids, gts, urls = \
        zip(*[(instance[product_id_column], instance[gt_column], instance[url_column]) for instance in read_res])
    product_ids, gts, urls = np.array(product_ids), np.array(gts), np.array(urls)
    if args.set_cats is not None:
        keeps = np.isin(gts, args.set_cats)
    elif args.set_date is not None:
        dates = np.array([res[args.date_column] for res in read_res])
        keeps = np.isin(dates, args.set_date)
    else:
        raise ValueError("Invalid set_cats or set_date!")
    assert keeps.shape[0] >= 1, "the sql data number must be greater than 0!"
    product_ids, gts, urls = product_ids[keeps], np.array(gts)[keeps], np.array(urls)[keeps]
    url_map = {}
    for idx, (product_id, gt, url) in enumerate(zip(product_ids, gts, urls)):
        product_id = product_id or f"{idx:34d}"
        url_map.update({product_id: {"gt": gt, "url": url}})
    print(f"save {len(url_map)} urls to test_{args.brand_id}.csv")
    save_dict2csv(url_map, f"test_{args.brand_id}.csv")


if __name__ == '__main__':
    args = parse_args()
    sql_server = create_sql_server(args.brand_id)
    run_sql_with_save(sql_server, args)
