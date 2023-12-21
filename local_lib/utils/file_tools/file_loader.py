import numpy as np


def load_data(file_path):
    with np.load(file_path) as data:
        # 从npz文件中获取数组
        feas = data["feats"].astype("float32")
        labels = data["gts"]
        fpaths = data["fpaths"]
    return feas, labels, fpaths


def load_csv_file(label_file):
    product_id_map = {}
    with open(label_file) as f:
        for line in f:
            try:
                id_record = line.strip().replace('"', "").split(",")
                product_id_map[id_record[0]] = id_record[1]
            except:
                print(f"line={line} is error!")
    return product_id_map
