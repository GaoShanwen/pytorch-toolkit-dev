import cv2
import os
import json
import numpy as np
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser("visualize boxes parameters")
    parser.add_argument("-i", "--img-root", type=str, required=True, default=None)
    parser.add_argument("-d", "--dst-root", type=str, required=True, default=None)
    parser.add_argument("-s", "--crop-size", type=int, default=640)
    parser.add_argument("-o", "--overlap", type=int, default=150)
    return parser.parse_args()


def do_crop(image_path, output_dir, crop_size, overlap):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    height, width, _ = image.shape

    label_path = image_path.replace(".jpg", ".txt")
    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f.readlines()]
    os.makedirs(output_dir, exist_ok=True)

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 遍历大图，裁剪子图和标签
    for y in range(0, height, crop_size - overlap):
        for x in range(0, width, crop_size - overlap):
            # 计算裁剪区域
            x_end = min(x + crop_size, width)
            y_end = min(y + crop_size, height)

            # 裁剪子图
            sub_image = image[y:y_end, x:x_end]
            sub_image_name = f"{image_name}_{x}_{y}.jpg"
            sub_image_path = os.path.join(output_dir, sub_image_name)
            cv2.imwrite(sub_image_path, sub_image)

            # 裁剪对应的 YOLO 标签
            sub_labels = []
            for label in labels:
                class_id, x_center, y_center, w, h = map(float, label)
                # 将 YOLO 坐标转换为像素坐标
                x_center_abs = x_center * width
                y_center_abs = y_center * height
                w_abs = w * width
                h_abs = h * height

                # 计算边界框的左上角和右下角坐标
                x1 = x_center_abs - w_abs / 2
                y1 = y_center_abs - h_abs / 2
                x2 = x_center_abs + w_abs / 2
                y2 = y_center_abs + h_abs / 2

                # 检查边界框是否在裁剪区域内
                if x1 < x_end and x2 > x and y1 < y_end and y2 > y:
                    # 计算裁剪后的边界框坐标
                    x1_crop = max(x1 - x, 0)
                    y1_crop = max(y1 - y, 0)
                    x2_crop = min(x2 - x, crop_size)
                    y2_crop = min(y2 - y, crop_size)

                    # 转换为 YOLO 格式
                    x_center_crop = (x1_crop + x2_crop) / 2 / crop_size
                    y_center_crop = (y1_crop + y2_crop) / 2 / crop_size
                    w_crop = (x2_crop - x1_crop) / crop_size
                    h_crop = (y2_crop - y1_crop) / crop_size

                    sub_labels.append(f"{int(class_id)} {x_center_crop} {y_center_crop} {w_crop} {h_crop}")

            # 保存裁剪后的标签
            if sub_labels:
                sub_label_name = f"{image_name}_{x}_{y}.txt"
                sub_label_path = os.path.join(output_dir, sub_label_name)
                with open(sub_label_path, 'w') as f:
                    f.write("\n".join(sub_labels))


if __name__ == '__main__':
    args = parse_args()
    dst_dirs = [root for root, _, _ in os.walk(args.img_root)]
    for dst_dir in dst_dirs:
        paths = [os.path.join(dst_dir, p) for p in os.listdir(dst_dir) if p.endswith(".jpg")]
        for image_path in paths:
            do_crop(image_path, args.dst_root, args.crop_size, args.overlap)
