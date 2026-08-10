"""
COCO 标注预处理脚本 v2

功能:
1. filter 模式: 保留 category_id >= 5 的类别，仅保留关键点索引 4-7
2. yolo2coco 模式: 将 YOLO 格式的姿态标注转为 RTMPose 支持的 COCO JSON 格式
   - 类别 5,6,7 -> 最终类别 0,1,2 -> 关键点 0,1,2,3
   - 类别 >=8    -> 最终类别 >=3   -> 关键点 4,5,6,7
   - 裁减区域中其他实例的关键点也保留在子图 annotation 中

用法 (yolo2coco):
    python data/scripts/pose/preprocess_cocov2.py \
        --mode yolo2coco \
        --dataset_dir data/pose-dataset/BakingRecognize \
        --train_file val.txt \
        --out_file data/pose-dataset/Bakingrefine/annotations/val.json \
        --expand_ratio 1.25 \
        --min_category_id 5
"""

import argparse
import json
import os
import yaml
from copy import deepcopy
from PIL import Image


def get_kp_range(class_id):
    """Get keypoint output range for a given class ID.

    All classes use original indices 3,4,5,6 (which contain valid keypoints).
    - Classes 5,6  -> output indices 0,1,2,3
    - Class 7      -> output indices 4,5,6,7
    - Classes > 7  -> output indices 8,9,10,11

    Returns:
        (out_start, out_end): Output keypoint range (exclusive end)
    """
    if class_id in (5, 6):
        return 0, 4
    if class_id == 7:
        return 4, 8
    return 8, 12


def get_skeleton_for_class(class_id):
    """Get skeleton for a given class ID (0-indexed format).

    Skeleton defines connections between keypoints for visualization.
    Each class has 4 keypoints that form a closed loop.
    - Classes 5,6  -> keypoints 0,1,2,3
    - Class 7      -> keypoints 4,5,6,7
    - Classes >7   -> keypoints 8,9,10,11
    """
    if class_id in (5, 6):
        return [[0, 1], [1, 2], [2, 3], [3, 0]]
    elif class_id == 7:
        return [[4, 5], [5, 6], [6, 7], [7, 4]]
    else:
        return [[8, 9], [9, 10], [10, 11], [11, 8]]


def filter_coco_annotations(
    ann_file: str,
    out_file: str,
    min_category_id: int = 5,
    kp_start_idx: int = 4,
    kp_end_idx: int = 7,
):
    """Filter COCO annotations by category and keypoint indices."""
    with open(ann_file, 'r') as f:
        coco = json.load(f)

    num_kp = kp_end_idx - kp_start_idx

    old_categories = coco.get('categories', [])
    old_cat_ids = {cat['id'] for cat in old_categories}
    kept_categories = [cat for cat in old_categories if cat['id'] >= min_category_id]
    kept_cat_ids = {cat['id'] for cat in kept_categories}

    cat_id_mapping = {}
    for new_idx, cat in enumerate(kept_categories):
        cat_id_mapping[cat['id']] = new_idx

    new_categories = []
    for cat in kept_categories:
        new_cat = deepcopy(cat)
        new_cat['id'] = cat_id_mapping[cat['id']]
        if 'keypoints' in new_cat:
            new_cat['keypoints'] = new_cat['keypoints'][kp_start_idx:kp_end_idx]
        if 'skeleton' in new_cat:
            old_skeleton = new_cat.get('skeleton', [])
            new_skeleton = []
            for edge in old_skeleton:
                e0, e1 = edge[0] - 1, edge[1] - 1
                if kp_start_idx <= e0 < kp_end_idx and kp_start_idx <= e1 < kp_end_idx:
                    new_skeleton.append([e0 - kp_start_idx + 1, e1 - kp_start_idx + 1])
            new_cat['skeleton'] = new_skeleton
        new_categories.append(new_cat)

    new_annotations = []
    filtered_ann_count = 0
    for ann in coco.get('annotations', []):
        cat_id = ann.get('category_id', -1)
        if cat_id not in kept_cat_ids:
            filtered_ann_count += 1
            continue
        new_ann = deepcopy(ann)
        new_ann['category_id'] = cat_id_mapping[cat_id]
        old_keypoints = ann.get('keypoints', [])
        new_keypoints = []
        for i in range(kp_start_idx, kp_end_idx):
            base_idx = i * 3
            new_keypoints.extend(old_keypoints[base_idx:base_idx + 3])
        new_ann['keypoints'] = new_keypoints
        new_ann['num_keypoints'] = num_kp
        new_annotations.append(new_ann)

    kept_image_ids = set(ann['image_id'] for ann in new_annotations)
    new_images = [img for img in coco.get('images', []) if img['id'] in kept_image_ids]

    new_coco = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'images': new_images,
        'annotations': new_annotations,
        'categories': new_categories,
    }

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(new_coco, f, indent=2)

    print(f"COCO Filter Summary: {len(new_annotations)} kept, {filtered_ann_count} filtered")


def expand_bbox(x, y, w, h, img_width, img_height, expand_ratio=1.25):
    """Expand bbox from center, keeping within image boundaries."""
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    new_w = w * expand_ratio
    new_h = h * expand_ratio
    crop_x = max(0.0, center_x - new_w / 2.0)
    crop_y = max(0.0, center_y - new_h / 2.0)
    crop_x_max = min(float(img_width), center_x + new_w / 2.0)
    crop_y_max = min(float(img_height), center_y + new_h / 2.0)
    return crop_x, crop_y, crop_x_max - crop_x, crop_y_max - crop_y


def yolo_to_coco(
    dataset_dir: str,
    train_file: str,
    out_file: str,
    dataset_yaml: str = None,
    sub_images_dir: str = None,
    expand_ratio: float = 1.25,
    min_category_id: int = 5,
):
    """Convert YOLO format multi-object pose labels to RTMPose single-object COCO JSON.

    - Classes 5,6,7 -> keypoints 0,1,2,3
    - Classes >=8    -> keypoints 4,5,6,7
    - Other instances' keypoints within the crop region are preserved.
    """
    import cv2

    if dataset_yaml is None:
        dataset_yaml = os.path.join(dataset_dir, 'dataset.yaml')

    with open(dataset_yaml, 'r') as f:
        ds_config = yaml.safe_load(f)

    class_names = ds_config.get('names', {})
    nc = ds_config.get('nc', len(class_names))
    kpt_shape = ds_config.get('kpt_shape', [7, 3])
    num_keypoints = kpt_shape[0]

    default_kpt_names = list(ds_config.get('kpt_names', {}).values())
    if not default_kpt_names:
        default_kpt_names = [f'keypoint_{i}' for i in range(num_keypoints)]

    TOTAL_KP = 12
    all_kpt_names = []
    for i in range(TOTAL_KP):
        if i < len(default_kpt_names):
            all_kpt_names.append(default_kpt_names[i])
        else:
            all_kpt_names.append(f'kp_{i}')

    if not os.path.isabs(train_file):
        train_file = os.path.join(dataset_dir, train_file)

    with open(train_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    if sub_images_dir is None:
        out_dir = os.path.dirname(out_file)
        sub_images_dir = os.path.join(out_dir, '..', 'images')
    sub_images_dir = os.path.abspath(sub_images_dir)
    os.makedirs(sub_images_dir, exist_ok=True)

    kept_class_ids = sorted([cid for cid in range(nc) if cid >= min_category_id])
    cat_id_mapping = {old_id: new_id for new_id, old_id in enumerate(kept_class_ids)}

    categories = []
    for old_id in kept_class_ids:
        cat_name = class_names.get(old_id, f'class_{old_id}')
        new_id = cat_id_mapping[old_id]
        categories.append({
            'id': new_id,
            'name': cat_name,
            'keypoints': all_kpt_names,
            'skeleton': get_skeleton_for_class(old_id),
        })

    images = []
    annotations = []
    image_id = 0
    ann_id = 0
    skipped_no_label = 0
    skipped_no_image = 0
    skipped_no_object = 0
    skipped_few_kpt = 0
    skipped_shape = 0
    total_objects = 0
    total_cropped = 0
    kpt_count_validated = False

    cat_original_counts = {cid: 0 for cid in range(nc)}
    cat_kept_counts = {new_id: 0 for new_id in cat_id_mapping.values()}

    labels_dir = os.path.join(dataset_dir, 'labels')
    images_dir = os.path.join(dataset_dir, 'images')

    for img_path in image_paths:
        if os.path.isabs(img_path):
            img_abs_path = img_path
        elif os.path.exists(img_path):
            img_abs_path = img_path
        elif os.path.exists(os.path.join(dataset_dir, img_path)):
            img_abs_path = os.path.join(dataset_dir, img_path)
        else:
            img_abs_path = img_path

        if not os.path.exists(img_abs_path):
            skipped_no_image += 1
            continue

        image = cv2.imread(img_abs_path)
        if image is None:
            skipped_no_image += 1
            continue

        img_height, img_width = image.shape[:2]

        rel_path = os.path.relpath(img_abs_path, images_dir)
        label_rel_path = os.path.splitext(rel_path)[0] + '.txt'
        label_abs_path = os.path.join(labels_dir, label_rel_path)

        if not os.path.exists(label_abs_path):
            skipped_no_label += 1
            continue

        # --- 第一遍：读取该图片的所有目标 ---
        all_objects = []
        with open(label_abs_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = list(map(float, line.split()))
                if len(values) < 5:
                    continue
                all_objects.append(values)

        # --- 第二遍：逐个目标生成子图 ---
        for obj_idx, values in enumerate(all_objects):
            class_id = int(values[0])
            if class_id < min_category_id:
                skipped_no_object += 1
                continue

            kpt_values = values[5:]
            actual_num_kpt = len(kpt_values) // 3
            if not kpt_count_validated:
                if actual_num_kpt != num_keypoints:
                    raise ValueError(
                        f"Keypoint count mismatch! Config says {num_keypoints}, "
                        f"label has {actual_num_kpt}. File: {label_abs_path}"
                    )
                kpt_count_validated = True

            cx, cy, bw, bh = values[1:5]
            abs_x = (cx - bw / 2.0) * img_width
            abs_y = (cy - bh / 2.0) * img_height
            abs_w = bw * img_width
            abs_h = bh * img_height

            cat_original_counts[class_id] = cat_original_counts.get(class_id, 0) + 1

            if abs_w > 0 and abs_h > 0:
                aspect_ratio = max(abs_w, abs_h) / min(abs_w, abs_h)
                if aspect_ratio > 3.0 or min(abs_w, abs_h) < 15.0:
                    skipped_shape += 1
                    continue

            crop_x, crop_y, crop_w, crop_h = expand_bbox(
                abs_x, abs_y, abs_w, abs_h, img_width, img_height, expand_ratio)

            if crop_w <= 0 or crop_h <= 0:
                skipped_no_object += 1
                continue

            x1, y1 = int(round(crop_x)), int(round(crop_y))
            x2, y2 = int(round(crop_x + crop_w)), int(round(crop_y + crop_h))
            sub_image = image[y1:y2, x1:x2]

            img_basename = os.path.splitext(os.path.basename(img_abs_path))[0]
            sub_img_name = f"{img_basename}_{x1}_{y1}_{class_id}.jpg"
            sub_img_path = os.path.join(sub_images_dir, sub_img_name)
            cv2.imwrite(sub_img_path, sub_image)

            images.append({
                'id': image_id,
                'file_name': sub_img_name,
                'width': int(crop_w),
                'height': int(crop_h),
            })

            # --- 构建 8 个关键点 ---
            keypoints = [0.0] * (TOTAL_KP * 3)

            # 填充当前目标的关键点
            # kpt_values 中原始索引 0-2 无效（v=0），有效数据从索引 3 开始
            # 输出索引 kp_start 到 kp_end-1 对应原始索引 3,4,5,6
            kp_start, kp_end = get_kp_range(class_id)
            for out_idx in range(kp_start, kp_end):
                orig_idx = 3 + (out_idx - kp_start)  # 原始索引 3,4,5,6
                base = orig_idx * 3
                if base + 2 < len(kpt_values):
                    kx = kpt_values[base] * img_width - crop_x
                    ky = kpt_values[base + 1] * img_height - crop_y
                    kv = int(kpt_values[base + 2])
                    keypoints[out_idx * 3:(out_idx + 1) * 3] = [kx, ky, kv]

            # 检查其他目标的关键点是否在裁减区域内
            for other_idx, other_values in enumerate(all_objects):
                if other_idx == obj_idx:
                    continue
                other_cls = int(other_values[0])
                if other_cls < min_category_id:
                    continue
                other_kp_start, other_kp_end = get_kp_range(other_cls)
                other_kpt_vals = other_values[5:]
                for out_idx in range(other_kp_start, other_kp_end):
                    orig_idx = 3 + (out_idx - other_kp_start)  # 原始索引 3,4,5,6
                    base = orig_idx * 3
                    if base + 2 >= len(other_kpt_vals):
                        continue
                    kx = other_kpt_vals[base] * img_width
                    ky = other_kpt_vals[base + 1] * img_height
                    kv = int(other_kpt_vals[base + 2])
                    if kv == 0:
                        continue
                    if crop_x <= kx < crop_x + crop_w and crop_y <= ky < crop_y + crop_h:
                        if keypoints[out_idx * 3 + 2] == 0:
                            keypoints[out_idx * 3:(out_idx + 1) * 3] = [
                                kx - crop_x, ky - crop_y, kv]

            num_visible = sum(1 for i in range(0, len(keypoints), 3) if keypoints[i + 2] > 0)

            if num_visible < 2:
                os.remove(sub_img_path)
                images.pop()
                skipped_few_kpt += 1
                continue

            annotations.append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id_mapping[class_id],
                'bbox': [abs_x - crop_x, abs_y - crop_y, abs_w, abs_h],
                'area': abs_w * abs_h,
                'keypoints': keypoints,
                'num_keypoints': num_visible,
                'iscrowd': 0,
            })

            ann_id += 1
            image_id += 1
            total_objects += 1
            total_cropped += 1
            cat_kept_counts[cat_id_mapping[class_id]] += 1

    coco_dict = {
        'info': {'description': f'YOLO to COCO v2: {train_file}', 'version': '2.0', 'year': 2026},
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }

    os.makedirs(os.path.dirname(out_file) if os.path.dirname(out_file) else '.', exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(coco_dict, f, indent=2)

    print(f"=" * 60)
    print(f"YOLO to COCO v2 Summary")
    print(f"=" * 60)
    print(f"Sub-images:    {total_cropped}")
    print(f"Annotations:   {total_objects}")
    print(f"Categories:    {len(categories)} (mapping: {cat_id_mapping})")
    print(f"Keypoints:     {TOTAL_KP} total (cls 5,6 -> 0-3; cls 7 -> 4-7; cls >7 -> 8-11)")
    print(f"Expand ratio:  {expand_ratio}")
    print(f"-" * 60)
    print(f"Per-category KP ranges:")
    for old_id in sorted(kept_class_ids):
        s, e = get_kp_range(old_id)
        print(f"  class {old_id}: keypoints [{s}, {e})")
    print(f"-" * 60)
    print(f"Skipped: image={skipped_no_image} label={skipped_no_label} "
          f"category={skipped_no_object} shape={skipped_shape} few_kpt={skipped_few_kpt}")
    print(f"-" * 60)
    for old_id in sorted(kept_class_ids):
        new_id = cat_id_mapping[old_id]
        name = class_names.get(old_id, f'class_{old_id}')
        total_count = cat_kept_counts.get(new_id, 0)
        
        kp_start, kp_end = get_kp_range(old_id)
        
        # 统计自身关键点范围内的可见关键点总数
        own_kpt_count = 0
        # 统计其他关键点范围内的可见关键点总数
        other_kpt_0_3 = 0  # 0-3 范围内的其他关键点
        other_kpt_4_7 = 0  # 4-7 范围内的其他关键点
        other_kpt_8_11 = 0  # 8-11 范围内的其他关键点
        
        for ann in annotations:
            if ann['category_id'] == new_id:
                kps = ann['keypoints']
                # 统计自身关键点范围内可见的关键点数量
                for i in range(kp_start, kp_end):
                    if kps[i*3+2] > 0:
                        own_kpt_count += 1
                
                # 统计其他关键点范围内的可见关键点数量
                for i in range(0, 12):
                    if i >= kp_start and i < kp_end:
                        continue  # 跳过自身关键点范围
                    if kps[i*3+2] > 0:
                        if 0 <= i <= 3:
                            other_kpt_0_3 += 1
                        elif 4 <= i <= 7:
                            other_kpt_4_7 += 1
                        elif 8 <= i <= 11:
                            other_kpt_8_11 += 1
        
        if total_count == 0:
            print(f"  [{old_id:2d} -> {new_id:2d}] {name:20s}: "
                  f"{cat_original_counts.get(old_id, 0):4d} -> {total_count:4d}")
            continue

        if old_id in (5, 6):
            parts = []
            if own_kpt_count > 0:
                parts.append(f"own_kp{kp_start}-{kp_end-1}: {own_kpt_count}")
            if other_kpt_4_7 > 0:
                parts.append(f"other_kp4-7: {other_kpt_4_7}")
            if other_kpt_8_11 > 0:
                parts.append(f"other_kp8-11: {other_kpt_8_11}")
            print(f"  [{old_id:2d} -> {new_id:2d}] {name:20s}: "
                  f"{cat_original_counts.get(old_id, 0):4d} -> {total_count:4d} ({', '.join(parts)})")
        elif old_id == 7:
            parts = []
            if own_kpt_count > 0:
                parts.append(f"own_kp{kp_start}-{kp_end-1}: {own_kpt_count}")
            if other_kpt_0_3 > 0:
                parts.append(f"other_kp0-3: {other_kpt_0_3}")
            print(f"  [{old_id:2d} -> {new_id:2d}] {name:20s}: "
                  f"{cat_original_counts.get(old_id, 0):4d} -> {total_count:4d} ({', '.join(parts)})")
        else:
            parts = []
            if own_kpt_count > 0:
                parts.append(f"own_kp{kp_start}-{kp_end-1}: {own_kpt_count}")
            if other_kpt_0_3 > 0:
                parts.append(f"other_kp0-3: {other_kpt_0_3}")
            if other_kpt_4_7 > 0:
                parts.append(f"other_kp4-7: {other_kpt_4_7}")
            print(f"  [{old_id:2d} -> {new_id:2d}] {name:20s}: "
                  f"{cat_original_counts.get(old_id, 0):4d} -> {total_count:4d} ({', '.join(parts)})")
    print(f"=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess COCO annotations v2')
    parser.add_argument('--mode', type=str, default='filter', choices=['filter', 'yolo2coco'])
    parser.add_argument('--ann_file', default=None)
    parser.add_argument('--min_category_id', type=int, default=5)
    parser.add_argument('--kp_start_idx', type=int, default=3)
    parser.add_argument('--kp_end_idx', type=int, default=7)
    parser.add_argument('--dataset_dir', default=None)
    parser.add_argument('--train_file', default=None)
    parser.add_argument('--dataset_yaml', default=None)
    parser.add_argument('--sub_images_dir', default=None)
    parser.add_argument('--expand_ratio', type=float, default=1.25)
    parser.add_argument('--out_file', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'yolo2coco':
        yolo_to_coco(
            dataset_dir=args.dataset_dir,
            train_file=args.train_file,
            out_file=args.out_file,
            dataset_yaml=args.dataset_yaml,
            sub_images_dir=args.sub_images_dir,
            expand_ratio=args.expand_ratio,
            min_category_id=args.min_category_id,
        )
    else:
        filter_coco_annotations(
            ann_file=args.ann_file,
            out_file=args.out_file,
            min_category_id=args.min_category_id,
            kp_start_idx=args.kp_start_idx,
            kp_end_idx=args.kp_end_idx,
        )