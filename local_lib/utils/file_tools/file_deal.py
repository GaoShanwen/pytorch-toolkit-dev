import os
import numpy as np
import pandas as pd


def print_acc_map(acc_map, csv_name):
    df = pd.DataFrame(acc_map).transpose()
    df.to_csv(csv_name)
    # print(df)


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
