import json
import os
import shutil
from pathlib import Path
from PIL import Image
import argparse


def yolo_to_coco(yolo_line, img_width, img_height):
    """Convert YOLO format to COCO bbox format."""
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


def create_coco_split(image_list_file, output_dir, prefix_map, split_name='train', categories=None):
    """Create one COCO split (train or val). Images go directly in split folder."""
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

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

        symlink_path = os.path.join(split_dir, img_filename)
        if not os.path.exists(symlink_path):
            os.symlink(img_full_path, symlink_path)

        coco_data["images"].append({
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
            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            ann_id += 1

        if img_id % 500 == 0:
            print(f"  Processed {img_id} images for {split_name}...")

    json_path = os.path.join(split_dir, '_annotations.coco.json')
    with open(json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    return len(coco_data["images"]), len(coco_data["annotations"])


def prepare_coco_dataset(
    dataset_yaml,
    output_dataset_dir,
    class_names,
    prefix_map=None
):
    """Prepare COCO format dataset from YOLO format dataset.yaml."""
    train_list = None
    val_list = None

    with open(dataset_yaml, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('train:'):
                train_file = line.split(':', 1)[1].strip().split('#')[0].strip()
                if train_file:
                    train_list = os.path.join(os.path.dirname(dataset_yaml), train_file)
            elif line.startswith('val:'):
                val_file = line.split(':', 1)[1].strip().split('#')[0].strip()
                if val_file:
                    val_list = os.path.join(os.path.dirname(dataset_yaml), val_file)

    if train_list is None:
        raise ValueError("Could not find train list in dataset.yaml")
    if val_list is None:
        raise ValueError("Could not find val list in dataset.yaml")

    categories = [{"id": idx, "name": name} for idx, name in enumerate(class_names)]

    print(f"Preparing COCO dataset at: {output_dataset_dir}")

    if train_list and os.path.exists(train_list):
        print(f"Processing train split ({train_list})...")
        train_imgs, train_anns = create_coco_split(train_list, output_dataset_dir, prefix_map, 'train', categories)
        print(f"  Train: {train_imgs} images, {train_anns} annotations")

    if val_list and os.path.exists(val_list):
        print(f"Processing val split ({val_list})...")
        val_imgs, val_anns = create_coco_split(val_list, output_dataset_dir, prefix_map, 'valid', categories)
        print(f"  Val: {val_imgs} images, {val_anns} annotations")

    print(f"\nConversion complete!")
    print(f"Dataset saved to: {output_dataset_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare COCO format dataset from YOLO format')
    parser.add_argument('--dataset_yaml', type=str,
                        default='/home/wenjie/workspace/pytorch-toolkit-dev/data/det-dataset/BakingRecognize/dataset.yaml',
                        help='Path to dataset.yaml file')
    parser.add_argument('--output', type=str,
                        default='/home/wenjie/workspace/pytorch-toolkit-dev/data/det-dataset/BakingRecognizeCOCO',
                        help='Output COCO dataset directory')
    parser.add_argument('--classes', type=str, nargs='+',
                        default=['other', 'donut', 'pineapplebun', 'multigrain', 'chia',
                                'Tray', 'Tray_Invalid', 'Tabletop', 'Oven_TopHandle',
                                'Oven_BottomHandle', 'Oven_TopInner', 'Oven_BottomInner',
                                'Grill', 'Screen_Number', 'Screen_Fuction'],
                        help='Class names in order')

    args = parser.parse_args()

    prefix_map = {
        'data/det-dataset/': '/Dataset/',
    }

    prepare_coco_dataset(
        dataset_yaml=args.dataset_yaml,
        output_dataset_dir=args.output,
        class_names=args.classes,
        prefix_map=prefix_map
    )