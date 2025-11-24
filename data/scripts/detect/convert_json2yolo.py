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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.format in ["box", "seg"], f"only support box or seg, you set is {args.format}"
    with open(os.path.join("data/det-dataset", args.task, "dataset.yaml"), 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        cats = [v for _, v in data["names"].items()]
    print("categories are: ", cats)
    target_num = 0
    dst_dirs = [root for root, _, _ in os.walk(args.src_root)]
    for video_dir in dst_dirs:
        for file_name in os.listdir(video_dir):
            if not file_name.endswith(".json"):
                continue
            file_path = os.path.join(video_dir, file_name)
            with open(file_path, 'r') as f:
                annos = json.load(f)
            img_h, img_w = annos.get("imageHeight", 0), annos.get("imageWidth", 0)
            obj_path = os.path.join(video_dir, file_name.replace('.json', '.txt'))
            with open(obj_path, 'w') as f:
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
                    x, y = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
                    w, h = (x2 - x1) / img_w, (y2 - y1) / img_h
                    f.write(f'{cat_id} {x} {y} {w} {h}\n')
                    target_num += 1

    print("object numer: ", target_num)
