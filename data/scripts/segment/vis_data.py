import cv2
import os
import numpy as np
import argparse
import random
import yaml
import colorsys
import random


def make_parser():
    parser = argparse.ArgumentParser("visualize boxes parameters")
    parser.add_argument("-s", "--src-files", type=str, required=True, default=None)
    parser.add_argument("-t", "--task", type=str, required=True, default=None)
    parser.add_argument("-f", "--format", type=str, default="det")
    parser.add_argument("-o", "--obj-root", type=str, default="vis_imgs")
    parser.add_argument("--k", type=int, default=1)
    return parser.parse_args()


def get_categories(args):
    with open(os.path.join('/'.join(args.src_files.split("/")[:2]), args.task, "dataset.yaml"), 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return [v for _, v in data["names"].items()]


def generate_colors(n, saturation=0.8, lightness=0.6, seed=None):
    if seed is not None:
        random.seed(seed)
    golden_ratio = 0.618
    hues = []
    for i in range(n):
        hues.append((random.random() + i * golden_ratio) % 1.0)
    colors = []
    for h in hues:
        r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
        colors.append( (int(r*255), int(g*255), int(b*255)) )
    return colors


def visualize_yolo_detection(img_path, obj_dir, categories=None, colors=None, thickness=2):
    anno_path = img_path.replace(".jpg", ".txt").replace(".jpeg", ".txt")
    if not os.path.exists(anno_path):
        return
    try:
        with open(anno_path, 'r') as f:
            boxes = [eval(line.strip().replace(' ', ',')) for line in f.readlines()]
    except:
        print(f"Error in parsing {anno_path}")
        return
    
    if not boxes:
        return
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    for box in boxes:
        l, x, y, w, h = box
        name = categories[args.task][l]
        color = colors[l]
        w, h = round(width * w), round(height * h)
        x, y = round(width * x - w // 2), round(height * y - h // 2)
    
        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    img_base = os.path.basename(img_path).split(".")[0]
    cv2.imwrite(os.path.join(obj_dir, f'vis_{img_base}.jpg'), img)


def visualize_yolo_segmentation(image_path, obj_dir, class_names=None, colors=None, thickness=2, alpha=0.7):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = image.shape[:2]
    overlay = image.copy()
    label_path = img_path.replace(".jpg", ".txt").replace(".jpeg", ".txt")
    if not os.path.exists(label_path):
        return
    with open(label_path, 'r') as f:
        lines = f.readlines()
    num = 0
    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 3 or (len(parts)-1) % 2 != 0:
            print(f"Skipping invalid line: {line.strip()}")
            continue

        class_id = int(parts[0])
        points_normalized = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        
        points = (points_normalized * np.array([[w, h]])).astype(np.int32)
        color = colors[class_id]

        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=thickness)

        if class_names:
            label = f"{class_names[class_id]}"
            text_pos = (points[0][0], points[0][1] - 10)
            cv2.putText(overlay, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        num += 1
    
    img = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    img_base = os.path.basename(image_path).split(".")[0]
    cv2.imwrite(os.path.join(obj_dir, f'vis_{img_base}.jpg'), img)


if __name__ == '__main__':
    args = make_parser()
    with open(args.src_files, 'r') as f:
        imgs = [line.strip() for line in f.readlines()]
    obj_dir = os.path.join(args.obj_root, args.task)
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)
    
    categories = get_categories(args)
    colors = generate_colors(len(categories))

    img_paths = random.sample(imgs, args.k)
    deal_func = visualize_yolo_segmentation if args.format == "seg" else visualize_yolo_detection
    for img_path in imgs: 
        deal_func(img_path, obj_dir, categories, colors)