import os
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 定义颜色字典
colors = [
    (255, 0, 0),  # 蓝色
    (0, 255, 0), # 绿色 
    (0, 165, 255),  # 橙色
    (0, 0, 255),  # 红色
    (0, 255, 255), # 
    (0, 255, 0), # 绿色 
    (255, 0, 0),  # 蓝色
    (0, 255, 0), # 绿色 
    (165, 255, 0),  # 橙色
    (165, 0, 255),  # 红色
    (165, 255, 255), # 
    (165, 125, 0), # 绿色 
    (255, 255, 255), # 绿色 
    (125, 125, 125), # 绿色 
    (165, 255, 0),  # 橙色
    (165, 0, 255),  # 红色
    (165, 255, 255), # 
    (165, 125, 0), # 绿色 
    (255, 255, 255), # 绿色 
    (125, 125, 125), # 绿色 
]


def vis_text(img, text, position, text_color, text_size, thickness=8, padding=0):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/simsun.ttc", text_size, encoding="utf-8")
    label_size = draw.textsize(text, fontText)
    x, y = position
    x1, y1 = x - padding, y - label_size[1] - padding
    x2, y2 = x + label_size[0] + padding, y + padding
    draw.rectangle([(x1, y1), (x2, y2)], fill=text_color[::-1])
    draw.text((x1, y1), text, (255, 255, 255), font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def map_dets2original(image_path, subimg_path, inf_path, obj_dir, labels):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    dst_path = os.path.join(obj_dir, f'{image_name}.jpg')
    image_path = dst_path if os.path.exists(dst_path) else image_path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    image_height, image_width, _ = image.shape

    sub_height, sub_width, _ = cv2.imread(subimg_path).shape
    # 读取 YOLO 标签
    with open(inf_path, 'r') as f:
        dets = [line.strip().split() for line in f.readlines()]

    x_min, y_min = map(int, subimg_path[:-4].split("_")[2:4])
    # print(x_min, y_min)
    mapped_dets = []
    for det in dets:
        class_id, x_center_sub, y_center_sub, width_sub, height_sub, score = map(float, det)
        if score < 0.5:
            continue
        # 转换坐标到子图像素坐标
        x_center_sub_abs = x_center_sub * sub_width
        y_center_sub_abs = y_center_sub * sub_height
        width_sub_abs = width_sub * sub_width
        height_sub_abs = height_sub * sub_height
        # 转换坐标到大图像素坐标
        x_center_abs = int(x_center_sub_abs + x_min)
        y_center_abs = int(y_center_sub_abs + y_min)
        width_abs = int(width_sub_abs)
        height_abs = int(height_sub_abs)

        x, y, w, h = x_center_abs-width_abs//2, y_center_abs-height_abs//2, width_abs, height_abs
        if int(class_id) not in [1, 2] or h >= 16:
            text = f"{labels[int(class_id)]} {round(score, 2)}"
            image = vis_text(image, text, (x, y), colors[int(class_id)], 32)
        else:
            label = ["TankUP", "TriangleWarning", "DangerWarning", "LicensePlate"][int(class_id)]
            text = f"{label} {round(score, 2)}"
            image = vis_text(image, text, (x, y), colors[int(class_id)], 32)
        cv2.rectangle(image, (x, y), (x + w, y + h), colors[int(class_id)], 2)
    cv2.imwrite(dst_path, image)


if __name__ == '__main__':
    # subimg_dir = "data/det-dataset/test/sub_imgs/dv2"
    # img_list = "runs/detect/exp6"
    # label_dir = "runs/detect/exp7/labels"
    output_dir = "data/det-dataset/test/result/dv1"
    os.makedirs(output_dir, exist_ok=True)
    subimg_dir = "data/det-dataset/test/sub_imgs/dv1"
    img_list = "runs/detect/exp10"
    label_dir = "runs/detect/exp11/labels"
    label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".txt")])
    # labels = ["TankUP", "TriangleWarning", "DangerWarning", "LicensePlate"]
    labels = ["TankUP", "危险", "危险品", "LicensePlate"]
    for label_path in tqdm(label_paths):
        image_name = label_path.replace(".txt", ".jpg")
        image_path = os.path.join(img_list, os.path.basename(image_name).split('_')[0]+'.jpg')
        subimg_path = os.path.join(subimg_dir, os.path.basename(image_name))
        map_dets2original(image_path, subimg_path, label_path, output_dir, labels)
