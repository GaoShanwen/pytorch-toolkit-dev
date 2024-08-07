######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.04.12
# filenaem: create_sql_data.py
# function: create csv file of sql data date=xxxx-xx-xx, brand_id=xxxx.
######################################################

from local_lib.utils.file_tools import save_dict2csv, create_sql_server, read_sql_data
from tools.eval_sql import parse_args


def run_sql_with_save(sql_server, args):
    if args.label_names_path is not None:
        label_map = {product_id: name for product_id, name in sql_server.read_names()}
        save_dict2csv(label_map, args.label_names_path, index=['name'])
        return

    product_ids, gts, urls, dates, preds = read_sql_data(sql_server, args.set_date, args.set_cats)
    assert gts.shape[0] >= 1, "the sql data number must be greater than 0!"
    url_map = {}
    for idx, (product_id, gt, url, date, pred) in enumerate(zip(product_ids, gts, urls, dates, preds)):
        product_id = product_id or f"{idx:34d}"
        url_map.update({product_id: {"gt": gt, "url": url, "date": date, "prediction": pred}})
    print(f"save {len(url_map)} urls to test_{args.brand_id}.csv")
    save_dict2csv(url_map, f"need_{args.brand_id}.csv")


if __name__ == '__main__':
    args = parse_args()
    sql_server = create_sql_server(args.brand_id, args.custom_keys)
    run_sql_with_save(sql_server, args)
