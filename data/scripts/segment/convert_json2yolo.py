import argparse
import json
import os
import numpy as np
import yaml


def make_parser():
    parser = argparse.ArgumentParser("visualize boxes parameters")
    parser.add_argument("-s", "--src-files", type=str, required=True, default=None)
    parser.add_argument("-t", "--task", type=str, required=True, default=None)
    parser.add_argument("-f", "--format", type=str, default="det")
    parser.add_argument("-o", "--obj-root", type=str, default="vis_imgs")
    parser.add_argument("--k", type=int, default=1)
    return parser.parse_args()


def labelme_to_yolo_seg(imgs_file, classes):
    with open(imgs_file, "r") as f:
        file_list = [file_path.strip() for file_path in f.readlines()]
    for json_path in file_list:
        json_path = json_path.replace(".jpg", ".json").replace(".jpeg", ".json")
        if not json_path.endswith(".json") or not os.path.exists(json_path):
            print(f"{json_path} is error!")
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_width, img_height = data['imageWidth'], data['imageHeight']
        txt_path = json_path.replace(".json", ".txt")
        with open(txt_path, 'w') as f:
            for shape in data['shapes']:
                label = shape["label"].replace("Desk", "desk").replace("Person","person")
                ploys = (np.array(shape['points']).reshape((-1,2)) * [1/img_width, 1/img_height]).reshape((-1)).tolist()
                if len(ploys) >= 6:
                    line = f"{classes.index(label)} " + " ".join(map("{:.6f}".format, ploys))
                f.writelines(f"{line}\n")


if __name__ == "__main__":
    args = make_parser()
    src_files = args.src_files  # Labelme JSON文件目录
    with open(os.path.join('/'.join(args.src_files.split("/")[:2]), args.task, "dataset.yaml"), 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        cats = [v for _, v in data["names"].items()]
    print("categories are: ", cats)
    labelme_to_yolo_seg(src_files, cats)


