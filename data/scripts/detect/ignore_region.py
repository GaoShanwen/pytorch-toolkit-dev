import json
import os
import argparse
import numpy as np
import yaml
import torch
import sys
sys.path.append('.')
from local_lib.data.utils import box_ioa


def parse_args():
    parser = argparse.ArgumentParser("visualize boxes parameters")
    parser.add_argument("-s", "--src-files", type=str, required=True, default=None)
    parser.add_argument("-t", "--task", type=str, required=True, default=None)
    parser.add_argument("-f", "--format", type=str, default='box')
    return parser.parse_args()


def nms(annos, threshold=0.999):
    if not len(annos):
        return []
    
    boxes = annos[:,1:]
    # 按分数降序排序的索引
    order = list(range(len(boxes))) #sorted(range(len(scores)), key=lambda i: -scores[i])
    selected = []
    while order:
        current = order.pop(0)
        selected.append(current)
        # 筛选IoU低于阈值的框
        new_order = []
        for idx in order:
            iou_val = iou(boxes[current], boxes[idx])
            if iou_val <= threshold:
                new_order.append(idx)
        order = new_order
    return [annos[i] for i in selected]


def iou(box1, box2):
    # 计算交集区域的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    # 计算各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # 计算并集面积
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


if __name__ == '__main__':
    args = parse_args()
    assert args.format in ["box", "seg"], f"only support box or seg, you set is {args.format}"
    with open(os.path.join('/'.join(args.src_files.split("/")[:2]), args.task, "dataset.yaml"), 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        cats = [v for _, v in data["names"].items()]
    
    print("categories are: ", cats)
    target_num = 0
    dst_dirs = [root for root, _, _ in os.walk(args.src_files)]
    data_infos = {cat: 0 for cat in cats}
    img_infos = data_infos.copy()
    replace_map = {"PersonMineCar": "VPRVanCar", "MineCar": "VGRMineCar", "MineCarRear": "VPRVanCar", "MineCarHead": "VGRMineCarHead"}

    with open(args.src_files, 'r') as f:
        imgs = [line.strip() for line in f.readlines()]
    ioav = 0.85
    for i, img_path in enumerate(imgs): 
        file_path = img_path.replace("images", "labels").replace(".jpg", ".json")
        if not os.path.exists(file_path):
            # print(file_path)
            continue
        with open(file_path, 'r') as f:
            annos = json.load(f)
        img_h, img_w = annos.get("imageHeight", 0), annos.get("imageWidth", 0)
        boxes = []
        for obj in annos.get("shapes", []):
            label_name = obj.get("label", None)
            label_name = replace_map[label_name] if label_name in replace_map else label_name
            if label_name not in cats:
                # print(f"label={label_name} passed!")
                continue
            cat_id = cats.index(label_name)
            # cat_id = min(cat_id, 12)
            data_infos[label_name] += 1
            if args.format == "seg":
                point = np.array(obj.get("points", [[]*4])).reshape((-1)).tolist()
                x1, y1, x2, y2 = min(point[::2]), min(point[1::2]), max(point[::2]), max(point[1::2])
            else:
                gap = 1 if annos.get("version", None) == "5.6.1" else 2
                [x1, y1], [x2, y2] = obj.get("points", [[]*4])[::gap]
            boxes.append([cat_id, x1, y1, x2, y2])
        if args.format != "seg":
            boxes = nms(np.array(boxes))
        if not len(boxes):
            continue
        boxes = np.hstack((np.array(boxes), np.zeros((len(boxes), 1))))
        v_idx = np.isin(boxes[:, 0], [12, 13])
        p_idx = boxes[:, 0] == 2
        if p_idx.sum() and v_idx.sum():
            p_keeps = (box_ioa(torch.from_numpy(boxes[p_idx, 1:5]).T, torch.from_numpy(boxes[v_idx, 1:5])) <= ioav).all(dim=0).numpy()
            ig_ids = np.ones((p_idx.sum()))
            ig_ids[p_keeps] = 0
            boxes[p_idx, -1] = ig_ids
            boxes = boxes.tolist()
        obj_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        if not len(boxes):
            print(obj_path)
        else:
            target_num += 1
        for cat_idx in set(np.array(boxes)[:, 0]):
            img_infos[cats[int(cat_idx)]] += 1
        with open(obj_path, 'w') as f:
            for box in boxes:
                cat_id, x1, y1, x2, y2, iscrowd = box
                x, y = (x1 + x2) / 2. / img_w, (y1 + y2) / 2 / img_h
                w, h = abs(x2 - x1) / img_w, abs(y2 - y1) / img_h 
                iscrowd = iscrowd or int(w < 0.00625 or h < 0.00625)
                w, h = max(0.00625, w), max(0.00625, h)
                # iscrowd = iscrowd or int(w < 0.015 or h < 0.03)
                # w, h = max(0.015, w), max(0.03, h)
                cat_id = min(cat_id, 12)
                f.write(f'{cat_id:.0f} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {iscrowd:.0f}\n')
                # target_num += 1

    print("object numer: ", target_num)
    print("data_infos: ", data_infos)
    print("img_infos: ", img_infos)
