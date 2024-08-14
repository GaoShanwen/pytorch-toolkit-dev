######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.04.12
# filenaem: create_sql_data.py
# function: create csv file of sql data date=xxxx-xx-xx, brand_id=xxxx.
######################################################
import argparse
from local_lib.utils.file_tools import save_dict2csv, create_sql_server, read_sql_data


def parse_args():
    parser = argparse.ArgumentParser(description="Create SQL Arguments")
    parser.add_argument("--instance-idx", type=int, default=0)
    parser.add_argument("--custom-keys", type=str, 
        default="sBarcode,sStoreId,sImgUrl,sTradeFlowNo,dCreateDate,src")
    parser.add_argument("--brand-id", type=int, default=1386)
    parser.add_argument("--set-date", type=str, nargs='*', default=None)
    parser.add_argument("--set-cats", type=str, nargs='*', default=None)
    parser.add_argument("--set-stores", type=str, nargs='*', default=None)
    parser.add_argument("--label-names-path", type=str, default=None)
    return parser.parse_args()


def run_sql_with_save(sql_server, args):
    if args.label_names_path is not None:
        label_map = {product_id: name for product_id, name in sql_server.read_names()}
        save_dict2csv(label_map, args.label_names_path, index=['name'])
        return

    product_ids, gts, store_ids, urls, dates, preds = read_sql_data(sql_server, args.set_date, args.set_cats, args.set_stores)
    assert gts.shape[0] >= 1, "the sql data number must be greater than 0!"
    url_map = {}
    for idx, (product_id, gt, store, url, date, pred) in enumerate(zip(product_ids, gts, store_ids, urls, dates, preds)):
        product_id = product_id or f"{idx:34d}"
        url_map.update({product_id: {"gt": gt, "url": url, "store": store, "date": date, "prediction": pred}})
    print(f"save {len(url_map)} urls to test_{args.brand_id}.csv")
    save_dict2csv(url_map, f"need_{args.brand_id}.csv")


if __name__ == '__main__':
    args = parse_args()
    sql_server = create_sql_server(args.brand_id, args.custom_keys)
    # sql_server = create_sql_server(args.brand_id)
    run_sql_with_save(sql_server, args)
