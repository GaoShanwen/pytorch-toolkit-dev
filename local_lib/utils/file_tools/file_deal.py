import os
import csv
import numpy as np
import pandas as pd
from typing import Union, List


def save_dict2csv(data, csv_name, index=None):
    df = pd.DataFrame(data, index).transpose()
    df.to_csv(csv_name)


def load_data(file_path):
    with np.load(file_path) as data:
        # 从npz文件中获取数组
        feas = data["feats"].astype("float32")
        labels = data["gts"]
        fpaths = data["fpaths"]
    return feas, labels, fpaths


def load_csv_file(
    label_file: str, 
    key_name: str="gt", 
    value_name: str="url", 
    to_int: bool=False, 
    frist_name: bool=False,
    concat_value: bool=False,
)->dict:
    product_id_map = {}
    with open(label_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            k, v = row[key_name], row[value_name]
            k = int(k) if to_int else k
            v = v.split("/")[0] if frist_name else v
            try:
                if concat_value:
                    if k not in product_id_map.keys():
                        product_id_map[row[key_name]] = []
                    product_id_map[k].append(v)
                else:
                    product_id_map.update({k: v})
            except:
                print(f"line={row} is error!")
    return product_id_map

def load_names(
        label_files: Union[str, List[str]], 
        idx_column: int=1, 
        name_column: int=2, 
        to_int: bool=True, 
        frist_name: bool=True,
    ):
    product_id_map = {}
    if isinstance(label_files, str):
        label_files = [label_files]
    for label_file in label_files:
        with open(label_file, "r") as f:
            for line in f.readlines():#[1:]:
                try:
                    id_record = line.strip().replace('"', "").split(",")
                    key = int(id_record[idx_column]) if to_int else id_record[idx_column]
                    value = id_record[name_column].split("/")[0] if frist_name else id_record[name_column]
                    product_id_map.update({key: value})
                except:
                    print(f"line={line.strip()} is passed!")
    return product_id_map


def save_keeps_file(labels, files, obj_files):
    with open(obj_files, "w") as f:
        for label, filename in zip(labels, files):
            f.write(f"{filename}, {label}\n")


def save_feat(feats, idx, save_dir="./output/features"):
    np.savez(f"{save_dir}/{idx:08d}.npz", feats=feats)  # , gts=gts.numpy())


def init_feats_dir(save_dir="./output/features"):
    if os.path.exists(save_dir):
        for file_name in os.listdir(save_dir):
            if not file_name.endswith(".npz"):
                raise f"{file_name} is error in {save_dir}!"
            os.remove(os.path.join(save_dir, file_name))
        os.removedirs(save_dir)
    os.makedirs(save_dir)


def merge_feat_files(load_dir="./output/features", infer_mode="val", file_infos=None, convert_map=None):
    files = sorted(os.listdir(load_dir))
    feats = []
    for file_name in files:
        file_path = os.path.join(load_dir, file_name)
        data = np.load(file_path)
        feats.append(data["feats"])
    merge_feats = np.concatenate(feats)
    merge_fpaths, merge_gts = zip(*(file_infos))
    merge_gts = np.array(merge_gts)
    if convert_map is not None:
        merge_gts = convert_map[merge_gts]
    np.savez(f"{load_dir}-{infer_mode}.npz", feats=merge_feats, gts=merge_gts, fpaths=merge_fpaths)
