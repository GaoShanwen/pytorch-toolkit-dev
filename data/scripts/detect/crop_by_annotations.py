import cv2
import os
import json
import numpy as np
import argparse
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser("visualize boxes parameters")
    parser.add_argument("-i", "--img-root", type=str, required=True, default=None)
    parser.add_argument("-d", "--dst-root", type=str, required=True, default=None)
    parser.add_argument("-s", "--crop-size", type=int, default=640)
    parser.add_argument("-o", "--overlap", type=int, default=150)
    return parser.parse_args()


def expand_bbox(bbox, image_width, image_height, expand_pixels=0, expand_ratio=1.0, postfix=[]):
    x_min, y_min, x_max, y_max = bbox

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    if expand_ratio != 1.0:
        width *= expand_ratio
        height *= expand_ratio

    x_min = max(0, int(center_x - width / 2 - expand_pixels))
    y_min = max(0, int(center_y - height / 2 - expand_pixels))
    x_max = min(image_width, int(center_x + width / 2 + expand_pixels))
    y_max = min(image_height, int(center_y + height / 2 + expand_pixels))

    return [x_min, y_min, x_max, y_max] + postfix


def crop_and_save(image_path, label_path, output_dir, expand_pixels, expand_ratio, cropped_id=0, inf_anno=False):
    """
    将大图中的汽车按检测框剪切成子图，并转换包含在其中的标签。

    参数:
        image_path (str): 大图路径。
        label_path (str): YOLO 标签路径。
        output_dir (str): 输出目录。
        car_class_id (int): 汽车类别的 ID（默认为 2，根据数据集调整）。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    image_height, image_width, _ = image.shape

    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    car_boxes = []
    for label in labels:
        class_id = int(label[0])
        if class_id != cropped_id:
            continue
        x_center, y_center, width, height = map(float, label[1:5])
        x_center_abs = x_center * image_width
        y_center_abs = y_center * image_height
        width_abs = width * image_width
        height_abs = height * image_height
        x_min = int(x_center_abs - width_abs / 2)
        y_min = int(y_center_abs - height_abs / 2)
        x_max = int(x_center_abs + width_abs / 2)
        y_max = int(y_center_abs + height_abs / 2)
        box = [x_min, y_min, x_max, y_max]
        s = [label[5]] if inf_anno else []
        car_boxes.append(expand_bbox(box, image_width, image_height, expand_pixels, expand_ratio, s))

    for i, box in enumerate(car_boxes):
        (x_min, y_min, x_max, y_max) = box[:4]
        postfix = "" if inf_anno else f"_{box[4]}"
        sub_image = image[y_min:y_max, x_min:x_max]
        sub_image_name = f"{image_name}_{x_min}_{y_min}{postfix}.jpg"
        sub_image_path = os.path.join(output_dir, sub_image_name)
        cv2.imwrite(sub_image_path, sub_image)

        if inf_anno:
            continue
        sub_labels = []
        for label in labels:
            class_id, x_center, y_center, width, height = map(float, label)
            if class_id == cropped_id:
                continue
            x_center_abs = x_center * image_width
            y_center_abs = y_center * image_height
            width_abs = width * image_width
            height_abs = height * image_height
            x1 = x_center_abs - width_abs / 2
            y1 = y_center_abs - height_abs / 2
            x2 = x_center_abs + width_abs / 2
            y2 = y_center_abs + height_abs / 2
            if x1 >= x_min and x2 <= x_max and y1 >= y_min and y2 <= y_max:
                x_center_crop = (x_center_abs - x_min) / (x_max - x_min)
                y_center_crop = (y_center_abs - y_min) / (y_max - y_min)
                width_crop = width_abs / (x_max - x_min)
                height_crop = height_abs / (y_max - y_min)
                sub_labels.append(f"{int(class_id)} {x_center_crop} {y_center_crop} {width_crop} {height_crop}")

        if sub_labels:
            sub_label_name = f"{image_name}_{i}.txt"
            sub_label_path = os.path.join(output_dir, sub_label_name)
            with open(sub_label_path, 'w') as f:
                f.write("\n".join(sub_labels))


if __name__ == '__main__':
    expand_ratio = 1.1
    expand_pixels = 0
    # output_dir = "data/det-dataset/dv-subimg/val"
    # img_list = "data/det-dataset/DargeVehicle/val.txt"
    # output_dir = "data/det-dataset/test/sub_imgs/dv2"
    # img_list = "data/det-dataset/test/imgs/dv2"
    # label_dir = "runs/detect/exp6/labels"
    output_dir = "data/det-dataset/test/sub_imgs/dv1"
    img_list = "data/det-dataset/test/imgs/dv1"
    label_dir = "runs/detect/exp10/labels"
    if img_list.endswith(".txt"):
        with open(img_list, "r") as f:
            image_paths = [line.strip() for line in f.readlines()]
    else:
        image_paths = [os.path.join(img_list, f) for f in os.listdir(img_list) if f.endswith(".jpg")]
    for image_path in tqdm(image_paths):
        label_path = image_path.replace(".jpg", ".txt")
        if label_dir:
            label_path = os.path.join(label_dir, os.path.basename(label_path))
        if not os.path.exists(label_path):
            continue
        crop_and_save(image_path, label_path, output_dir, expand_pixels, expand_ratio, cropped_id=0, inf_anno=True)
