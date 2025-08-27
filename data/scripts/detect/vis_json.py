import cv2
import os
import json
import numpy as np
import argparse
import random
from tkinter import _flatten
import yaml
import colorsys


def make_parser():
    parser = argparse.ArgumentParser("visualize boxes parameters")
    parser.add_argument("-s", "--src-files", type=str, required=True, default=None)
    parser.add_argument("-t", "--task", type=str, required=True, default=None)
    parser.add_argument("-o", "--obj-root", type=str, default="vis_imgs")
    parser.add_argument("--k", type=int, default=1)
    return parser.parse_args()


def generate_colors(n, saturation=0.8, lightness=0.6, seed=None):
    if seed is not None:
        random.seed(seed)
    
    golden_ratio = 0.618033988749895 # 黄金角分割法生成色相
    hues = []
    for i in range(n):
        hues.append((random.random() + i * golden_ratio) % 1.0)
    colors = []
    for h in hues:
        r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
        colors.append( (int(r*255), int(g*255), int(b*255)) )
    return colors


if __name__ == '__main__':
    args = make_parser()
    with open(os.path.join("data/det-dataset", args.task, "dataset.yaml"), 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        categories = [v for _, v in data["names"].items()]
    print("categories are: ", categories)
    colors = generate_colors(len(categories))
    with open(args.src_files, 'r') as f:
        imgs = [line.strip() for line in f.readlines()]
    obj_dir = os.path.join(args.obj_root, args.task)
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)
    imgs = random.sample(imgs, args.k)
    for img_path in imgs:
        anno_path = img_path.replace(".jpg", ".json")
        if not os.path.exists(anno_path):
            continue
        # try:
        with open(anno_path) as f:
            data = json.load(f)  # 解析 JSON 文件
        if data["version"] == "5.6.1":
            boxes = [[d["label"]]+list(_flatten(d["points"])) for d in data["shapes"]]
        else:
            boxes = [[d["label"]]+list(_flatten(d["points"][::2])) for d in data["shapes"]]
        # boxes = [b for b in boxes if b[0] == 3]
        if not boxes:
            continue
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        for box in boxes:
            name, x1, y1, x2, y2 = box
            x1, y1, x2, y2 = [round(i) for i in [x1, y1, x2, y2]]
            # print(img_path, name)
            color = colors[categories.index(name)]
            # w, h = round(width * w), round(height * h)
            # x, y = round(width * x - w / 2), round(height * y - h / 2)
            try:
                cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except:
                import pdb; pdb.set_trace()
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        img_base = os.path.basename(img_path).split(".")[0]
        # print(img_path, img_base)
        cv2.imwrite(os.path.join(obj_dir, f'vis_{img_base}.jpg'), img)
