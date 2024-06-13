import os
import multiprocessing

import wget
import tqdm


def load_csv_file(label_file: str, idx_column: int=1, url_column: int=3):
    product_id_map = {}
    with open(label_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            try:
                id_record = line.strip().replace('"', "").split(",")
                if id_record[idx_column] not in product_id_map.keys():
                    product_id_map[id_record[idx_column]] = []
                product_id_map[id_record[idx_column]].append(id_record[url_column])
            except:
                print(f"line={line} is error!")
    return product_id_map, len(lines)-1


def copy_file(file_info):
    ori_path, dst_path = file_info
    try:
        wget.download(ori_path, dst_path, bar=False)
    except:
        # print(f"{ori_path} is error!")
        pass
 

def main(download_urls, dst_root):
    res, copyed_num = load_csv_file(download_urls, idx_column=1, url_column=2)
    result_list_tqdm = []
    file_infos = []
    for i, (cat, urls) in enumerate(res.items()):
        obj_dir = os.path.join(dst_root, cat)
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)
        for url in urls:
            file_infos.append([url, obj_dir])
    
    with multiprocessing.Pool(processes=12) as pool:
        list(tqdm.tqdm(pool.imap(copy_file, file_infos), total=copyed_num))


if __name__ == "__main__":
    brand_id = 1438 #1267 #1386
    urls_file = f"test_{brand_id}.csv"
    obj_root = f"dataset/function_test/package_way/{brand_id}"
    if not os.path.exists(obj_root):
        os.makedirs(obj_root)

    main(urls_file, obj_root)

