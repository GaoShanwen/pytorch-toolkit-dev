import json
import os
import argparse
import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser("visualize boxes parameters")
    parser.add_argument("-s", "--src-root", type=str, required=True, default=None)
    parser.add_argument("-t", "--task", type=str, required=True, default=None)
    parser.add_argument("-f", "--format", type=str, default='box')
    # parser.add_argument("-o", "--obj-root", type=str, default="vis_imgs")
    # parser.add_argument("--k", type=int, default=1)
    return parser.parse_args()


def nms(boxes, threshold):
    if not boxes:
        return []
    
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
    return [boxes[i] for i in selected]


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
    with open(os.path.join("data/det-dataset", args.task, "dataset.yaml"), 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        cats = [v for _, v in data["names"].items()]
    print("categories are: ", cats)
    # cats = categories.get(args.task, [])
    target_num = 0
    dst_dirs = [root for root, _, _ in os.walk(args.src_root)]
    # for video_name in os.listdir(args.src_root):
    #     video_dir = os.path.join(args.src_root, video_name)
    for video_dir in dst_dirs:
        for file_name in os.listdir(video_dir):
            if not file_name.endswith(".json"):
                continue
            file_path = os.path.join(video_dir, file_name)
            with open(file_path, 'r') as f:
                annos = json.load(f)
            
            img_h, img_w = annos.get("imageHeight", 0), annos.get("imageWidth", 0)
            obj_path = os.path.join(video_dir, file_name.replace('.json', '.txt'))
            boxes = []
            for obj in annos.get("shapes", []):
                label_name = obj.get("label", None)
                if label_name not in cats:
                    continue
                cat_id = cats.index(label_name)
                if args.format == "seg":
                    point = np.array(obj.get("points", [[]*4])).reshape((-1)).tolist()
                    x1, y1, x2, y2 = min(point[::2]), min(point[1::2]), max(point[::2]), max(point[1::2])
                else:
                    gap = 1 if annos.get("version", None) == "5.6.1" else 2
                    [x1, y1], [x2, y2] = obj.get("points", [[]*4])[::gap]
                boxes.append([x1, y1, x2, y2, cat_id])
            # boxes = nms(boxes, 0.1)
            with open(obj_path, 'w') as f:
                for box in boxes:
                    x1, y1, x2, y2, cat_id = box
                    if cat_id != 4:
                        continue
                    x, y = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
                    w, h = (x2 - x1) / img_w, (y2 - y1) / img_h
                    f.write(f'0 {x} {y} {w} {h}\n')
                    target_num += 1

    print("object numer: ", target_num)
