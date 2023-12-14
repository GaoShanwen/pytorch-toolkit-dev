######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: mongo_client.py
# function: read file from mongo.
######################################################
import argparse
import os
import tqdm
import numpy as np
from urllib import parse
from shutil import copyfile

import pymongo


def connect_mongodb():
    # mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    # server = '120.92.86.215'
    # server = '172.17.0.1'
    server = "10.0.2.47"
    port = "27017"
    user = parse.quote_plus("RXTech")
    pwd = parse.quote_plus("ZMKM@Retail01")
    uri = "mongodb://{0}:{1}@{2}:{3}/".format(user, pwd, server, port)
    mongo_client = pymongo.MongoClient(uri, connect=False)

    return mongo_client


def sync_image(image_path):
    dst_file = os.path.join("/data/AI-scales/images/", image_path)
    if not os.path.exists(dst_file):
        src_file = os.path.join("/DATA/disk1/AI-scales/images/", image_path)
        if not os.path.exists(src_file):
            print(f"Missing file: {src_file}")
            return False
        else:
            print(f"Copy file: {src_file} {dst_file}")
            dst_dir, _ = os.path.split(dst_file)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            copyfile(src_file, dst_file)
        img = cv2.imread(dst_file)
        try:
            img.shape
        except:
            os.remove(dst_file)
            print(dst_file, "broken image")
            return False
        if img.shape[2] != 3:
            os.remove(dst_file)
            print(dst_file, "none rgb image")
            return False
    return True


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--brand_id", type=str, default="0")
    parser.add_argument("--label_file", type=str, default="dataset/zero_dataset/label_names.csv")
    parser.add_argument("--save_root", type=str, default="dataset/removeredundancy2")

    return parser.parse_args()


def main():
    # step 1: parse args
    args = parse_args()

    product_id_map = {}
    with open(args.label_file) as f:
        for line in f:
            id_record = line.strip().replace('"', "").split(",")
            product_id_map[int(id_record[0])] = id_record[1]

    # step 3: calculate
    mongo_client = connect_mongodb()
    AIScale = mongo_client["AIScale"]
    brand_id = args.brand_id
    save_root = args.save_root
    # ImageSet = AIScale['OrderSet']

    # step 3.2: calculate
    # num_class = len(product_id_map)
    data_dir = "/data/AI-scales/images/"

    def save_files(ImageSet, save_path, data_dir, brand_id, product_id_map):
        count = 0
        images = ImageSet.find({"brand_id": brand_id}, no_cursor_timeout=True)
        pbar = tqdm.tqdm(total=ImageSet.count_documents({"brand_id": brand_id}))
        with open(save_path, "w") as f:
            for data_record in images:
                pbar.update(1)
                product_id = data_record["product_id"]
                if int(product_id) not in product_id_map.keys():
                    continue
                image_path = os.path.join(data_dir, data_record["image_path"])
                if not os.path.exists(image_path):
                    continue
                try:
                    with open(image_path, "rb") as img_f:
                        img_f.seek(-2, 2)
                        if not img_f.read() == b"\xff\xd9":
                            continue
                except IOError:
                    continue
                f.write(f"{image_path},{product_id}\n")
                count += 1
            pbar.close()
            num_img = ImageSet.count_documents({"brand_id": brand_id})  # , 'type':1
            print(f"save num: {count}, origin num: {num_img}")

    TrainValSet = AIScale["TrainValSet"]  # ; TrainValSet.count_documents({'brand_id':brand_id})
    trainval_files = os.path.join(save_root, "trainval.txt")
    save_files(TrainValSet, trainval_files, data_dir, brand_id, product_id_map)
    ValSet = AIScale["ValSet"]
    val_files = os.path.join(save_root, "val.txt")
    save_files(ValSet, val_files, data_dir, brand_id, product_id_map)
    train_files = os.path.join(save_root, "train.txt")

    def choose_trainset(src_files, check_files, obj_files):
        with open(check_files, "r") as f:
            check_lines = np.array([line.strip() for line in f.readlines()])
        with open(src_files, "r") as f:
            src_lines = np.array([line.strip() for line in f.readlines()])

        masks = np.isin(src_lines, check_lines)
        save_lines = src_lines[~masks]
        with open(obj_files, "w") as f:
            for lines in tqdm.tqdm(save_lines):
                f.write(f"{lines}\n")
        print(f"save {len(save_lines)} imgs in {obj_files}")

    choose_trainset(trainval_files, val_files, train_files)


if __name__ == "__main__":
    main()
