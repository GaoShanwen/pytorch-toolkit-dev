import json
import os
from pathlib import Path
from PIL import Image
import argparse


def yolo_to_coco(yolo_line, img_width, img_height):
    """Convert YOLO format to COCO bbox format.

    YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized)
    COCO format: [x, y, width, height] (absolute pixels, x,y is top-left corner)
    """
    parts = yolo_line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])

    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height

    x = x_center_abs - width_abs / 2
    y = y_center_abs - height_abs / 2

    return [x, y, width_abs, height_abs], class_id


def convert_yolo_to_coco(
    image_list_file,
    output_json_path,
    class_names,
    prefix_map=None
):
    """Convert YOLO format annotations to COCO format JSON.

    Args:
        image_list_file: Path to file containing list of image paths
        output_json_path: Output path for COCO JSON
        class_names: List of class names in order
        prefix_map: Dict mapping path prefixes to replace, e.g. {'data/': '/Dataset/'}
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for idx, class_name in enumerate(class_names):
        coco_format["categories"].append({
            "id": idx,
            "name": class_name
        })

    with open(image_list_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    ann_id = 1
    for img_id, img_path in enumerate(image_paths, start=1):
        img_full_path = img_path
        if prefix_map:
            for old_prefix, new_prefix in prefix_map.items():
                if img_path.startswith(old_prefix):
                    img_full_path = img_path.replace(old_prefix, new_prefix)
                    break

        if not os.path.exists(img_full_path):
            print(f"Warning: Image not found: {img_full_path}")
            continue

        img = Image.open(img_full_path)
        img_width, img_height = img.size

        img_filename = os.path.basename(img_path)
        label_path = img_path.replace('/images/', '/labels-det/')
        label_path = label_path.replace('.jpg', '.txt')
        label_path = label_path.replace('.jpeg', '.txt')
        label_path = label_path.replace('.png', '.txt')

        if prefix_map:
            for old_prefix, new_prefix in prefix_map.items():
                if label_path.startswith(old_prefix):
                    label_path = label_path.replace(old_prefix, new_prefix)
                    break

        if not os.path.exists(label_path):
            print(f"Warning: Label not found: {label_path}")
            continue

        coco_format["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "width": img_width,
            "height": img_height
        })

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if not line.strip():
                continue
            bbox, class_id = yolo_to_coco(line, img_width, img_height)
            coco_format["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            ann_id += 1

        if img_id % 100 == 0:
            print(f"Processed {img_id} images...")

    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f"Conversion complete! Output saved to {output_json_path}")
    print(f"Total images: {len(coco_format['images'])}")
    print(f"Total annotations: {len(coco_format['annotations'])}")
    print(f"Total categories: {len(coco_format['categories'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert YOLO format to COCO format')
    parser.add_argument('--val_list', type=str,
                        default='data/det-dataset/BakingRecognize/val-det0811.txt',
                        help='Path to validation image list file')
    parser.add_argument('--output', type=str,
                        default='data/det-dataset/BakingRecognizeCOCO/valid/_annotations.coco.json',
                        help='Output COCO JSON path')
    parser.add_argument('--classes', type=str, nargs='+',
                        default=['other', 'donut', 'pineapplebun', 'multigrain', 'chia',
                                'Tray', 'Tray_Invalid', 'Tabletop', 'Oven_TopHandle',
                                'Oven_BottomHandle', 'Oven_TopInner', 'Oven_BottomInner',
                                'Grill', 'Screen_Number', 'Screen_Fuction'],
                        help='Class names in order')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    prefix_map = {
        'data/det-dataset/': '/Dataset/',
    }

    convert_yolo_to_coco(
        image_list_file=args.val_list,
        output_json_path=args.output,
        class_names=args.classes,
        prefix_map=prefix_map
    )