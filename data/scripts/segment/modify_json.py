import argparse
import json
import os
import numpy as np
import cv2


def make_parser():
    parser = argparse.ArgumentParser("visualize boxes parameters")
    parser.add_argument("-s", "--src-files", type=str, required=True, default=None)
    parser.add_argument("-t", "--task", type=str, required=True, default=None)
    parser.add_argument("-f", "--format", type=str, default="det")
    parser.add_argument("-o", "--obj-root", type=str, default="vis_imgs")
    parser.add_argument("--k", type=int, default=1)
    return parser.parse_args()


def labelme_to_yolo_seg(imgs_file):
    with open(imgs_file, "r") as f:
        file_list = [file_path.strip() for file_path in f.readlines()]
    for json_path in file_list:
        json_path = json_path.replace(".jpg", ".json").replace(".jpeg", ".json")
        if not json_path.endswith(".json") or not os.path.exists(json_path):
            print(f"{json_path} is error!")
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for i in range(len(data['shapes'])):
            data["shapes"][i]["label"] = data["shapes"][i]["label"].split('-')[0].replace("aihelmet", "helmet").replace("kongzhitai", "desk")
        data["imageData"] = None
            
        json.dump(data, open(json_path, 'w'))
        # break


if __name__ == "__main__":
    args = make_parser()
    src_files = args.src_files  # Labelme JSON文件目录

    labelme_to_yolo_seg(src_files)


