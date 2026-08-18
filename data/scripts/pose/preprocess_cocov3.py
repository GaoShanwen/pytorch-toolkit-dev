"""
COCO 标注预处理脚本

功能:
1. filter 模式: 保留 category_id >= 5 的类别，仅保留关键点索引 4-7
2. yolo2coco 模式: 将 YOLO 格式的姿态标注转为 RTMPose 支持的 COCO JSON 格式

用法 (filter):
    python data/scripts/pose/preprocess_coco.py \
        --mode filter \
        --ann_file data/pose-dataset/Person/annotations/train.json \
        --out_file data/pose-dataset/Person/annotations/train_filtered.json \
        --min_category_id 5 \
        --kp_start_idx 4 \
        --kp_end_idx 7

用法 (yolo2coco):
    python data/scripts/pose/preprocess_coco.py \
        --mode yolo2coco \
        --dataset_dir data/pose-dataset/BakingRecognize \
        --train_file val.txt \
        --out_file data/pose-dataset/Bakingrefine/annotations/val.json \
        --expand_ratio 1.25 \
        --min_category_id 5 \
        --kp_start_idx 4 \
        --kp_end_idx 7
"""

import argparse
import json
import os
import yaml
from copy import deepcopy
from PIL import Image


def filter_coco_annotations(
    ann_file: str,
    out_file: str,
    min_category_id: int = 5,
    kp_start_idx: int = 4,
    kp_end_idx: int = 7,
):
    """Filter COCO annotations by category and keypoint indices.

    Args:
        ann_file: Path to the input COCO JSON annotation file.
        out_file: Path to save the filtered COCO JSON.
        min_category_id: Minimum category ID to keep (inclusive).
        kp_start_idx: Starting index of keypoints to keep (0-indexed).
        kp_end_idx: Ending index of keypoints to keep (exclusive).
    """
    with open(ann_file, 'r') as f:
        coco = json.load(f)

    num_kp = kp_end_idx - kp_start_idx

    # 1. Filter categories: keep only those with id >= min_category_id
    old_categories = coco.get('categories', [])
    old_cat_ids = {cat['id'] for cat in old_categories}

    kept_categories = [cat for cat in old_categories if cat['id'] >= min_category_id]
    kept_cat_ids = {cat['id'] for cat in kept_categories}

    # Build old_id -> new_id mapping
    cat_id_mapping = {}
    for new_idx, cat in enumerate(kept_categories):
        cat_id_mapping[cat['id']] = new_idx

    # Update categories: remap id and update keypoints
    new_categories = []
    for cat in kept_categories:
        new_cat = deepcopy(cat)
        new_cat['id'] = cat_id_mapping[cat['id']]
        # Update keypoints info if present
        if 'keypoints' in new_cat:
            new_cat['keypoints'] = new_cat['keypoints'][kp_start_idx:kp_end_idx]
        if 'skeleton' in new_cat:
            # Filter skeleton: keep edges where both endpoints are in [kp_start_idx, kp_end_idx)
            old_skeleton = new_cat.get('skeleton', [])
            new_skeleton = []
            for edge in old_skeleton:
                e0, e1 = edge[0] - 1, edge[1] - 1  # COCO skeleton is 1-indexed
                if kp_start_idx <= e0 < kp_end_idx and kp_start_idx <= e1 < kp_end_idx:
                    new_edge = [e0 - kp_start_idx + 1, e1 - kp_start_idx + 1]
                    new_skeleton.append(new_edge)
            new_cat['skeleton'] = new_skeleton
        new_categories.append(new_cat)

    # 2. Filter annotations: keep only those with category_id in kept_cat_ids
    new_annotations = []
    filtered_ann_count = 0
    kept_ann_count = 0

    for ann in coco.get('annotations', []):
        cat_id = ann.get('category_id', -1)
        if cat_id not in kept_cat_ids:
            filtered_ann_count += 1
            continue

        new_ann = deepcopy(ann)
        # Remap category_id
        new_ann['category_id'] = cat_id_mapping[cat_id]

        # Extract keypoints: indices kp_start_idx to kp_end_idx-1
        old_keypoints = ann.get('keypoints', [])
        new_keypoints = []
        for i in range(kp_start_idx, kp_end_idx):
            base_idx = i * 3
            new_keypoints.extend(old_keypoints[base_idx:base_idx + 3])

        new_ann['keypoints'] = new_keypoints
        new_ann['num_keypoints'] = num_kp

        # Filter bbox (keep as is)
        kept_ann_count += 1
        new_annotations.append(new_ann)

    # 3. Filter images: keep only images referenced by kept annotations
    kept_image_ids = set(ann['image_id'] for ann in new_annotations)
    new_images = [img for img in coco.get('images', []) if img['id'] in kept_image_ids]

    # 4. Build new COCO dict
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

    # Print summary
    print(f"=" * 60)
    print(f"COCO Annotation Preprocessing Summary")
    print(f"=" * 60)
    print(f"Input:  {ann_file}")
    print(f"Output: {out_file}")
    print(f"-" * 60)
    print(f"Categories:")
    print(f"  Original: {len(old_categories)} (ids: {sorted(old_cat_ids)})")
    print(f"  Kept:     {len(new_categories)} (ids >= {min_category_id})")
    print(f"  Mapping:  {cat_id_mapping}")
    print(f"-" * 60)
    print(f"Keypoints:")
    print(f"  Original: {len(old_keypoints) // 3 if old_keypoints else 'N/A'} keypoints")
    print(f"  Kept:     {num_kp} keypoints (indices {kp_start_idx}-{kp_end_idx - 1})")
    print(f"-" * 60)
    print(f"Annotations: {len(new_annotations)} kept, {filtered_ann_count} filtered")
    print(f"Images:      {len(new_images)} kept")
    print(f"=" * 60)


def expand_bbox(x, y, w, h, img_width, img_height, expand_ratio=1.25):
    """Expand bbox from center, keeping within image boundaries.

    Args:
        x, y: Top-left corner (absolute pixels).
        w, h: Width and height (absolute pixels).
        img_width, img_height: Image dimensions.
        expand_ratio: Expansion ratio (default 1.25).

    Returns:
        (crop_x, crop_y, crop_w, crop_h): Expanded bbox in absolute pixels.
    """
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    new_w = w * expand_ratio
    new_h = h * expand_ratio

    crop_x = max(0.0, center_x - new_w / 2.0)
    crop_y = max(0.0, center_y - new_h / 2.0)
    crop_x_max = min(float(img_width), center_x + new_w / 2.0)
    crop_y_max = min(float(img_height), center_y + new_h / 2.0)

    crop_w = crop_x_max - crop_x
    crop_h = crop_y_max - crop_y

    return crop_x, crop_y, crop_w, crop_h


def convex_hull_area(points):
    """Area of the convex hull of a set of 2D points (Andrew's monotone chain).

    Args:
        points: Iterable of (x, y) tuples. Must contain >= 3 distinct points.

    Returns:
        Convex hull area. Returns 0 if fewer than 3 points or all points are collinear.
    """
    pts = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0

    area = 0.0
    n = len(hull)
    for i in range(n):
        j = (i + 1) % n
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    return abs(area) * 0.5


def is_triangle_like(points, min_ratio=0.92):
    """Check whether 4 (or more) points lie approximately on a single triangle.

    For each combination of 3 points, the triangle area is computed; the largest
    one is compared against the convex hull area of all points. When the ratio
    `max_triangle_area / convex_hull_area` exceeds `min_ratio`, the points are
    considered to be (approximately) a triangle — i.e. one point lies on or
    inside the triangle formed by the other three.

    Args:
        points: Iterable of (x, y) tuples. Must contain >= 3 points.
        min_ratio: Threshold in (0, 1]. Default 0.92 allows a small numerical margin.

    Returns:
        (is_triangle: bool, max_triangle_area: float, hull_area: float, ratio: float)
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    if n < 3:
        return False, 0.0, 0.0, 0.0

    hull_area = convex_hull_area(pts)
    if hull_area <= 0:
        return False, 0.0, 0.0, 0.0

    max_tri = 0.0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                a = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) * 0.5
                if a > max_tri:
                    max_tri = a

    ratio = max_tri / hull_area if hull_area > 0 else 0.0
    return ratio >= min_ratio, max_tri, hull_area, ratio


def yolo_to_coco(
    dataset_dir: str,
    train_file: str,
    out_file: str,
    dataset_yaml: str = None,
    sub_images_dir: str = None,
    expand_ratio: float = 1.25,
    min_category_id: int = 5,
    kp_start_idx: int = 4,
    kp_end_idx: int = 8,
    triangle_out_dir: str = None,
    triangle_min_ratio: float = 0.92,
):
    """Convert YOLO format multi-object pose labels to RTMPose single-object COCO JSON.

    For each annotation with category_id >= min_category_id:
    1. Crop the bbox region from the original image with expansion.
    2. Save the cropped sub-image.
    3. Adjust keypoints (indices kp_start_idx to kp_end_idx-1) relative to the crop.
    4. Remap category_id to 0-based index.

    YOLO label format (per line):
        class_id cx cy w h kpt1_x kpt1_y kpt1_v ... kptN_x kptN_y kptN_v

    All coordinates are normalized to [0, 1].
    Visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible.

    Args:
        dataset_dir: Root directory of the dataset.
        train_file: Path to the train list file.
        out_file: Path to save the output COCO JSON file.
        dataset_yaml: Path to dataset.yaml (default: dataset_dir/dataset.yaml).
        sub_images_dir: Directory to save cropped sub-images.
                        Default: same directory level as out_file, named 'images'.
        expand_ratio: Bbox expansion ratio (default 1.25).
        min_category_id: Minimum category ID to keep (default 5).
        kp_start_idx: Start index of keypoints to keep (default 4).
        kp_end_idx: End index of keypoints to keep, exclusive (default 8).
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
    skeleton = ds_config.get('skeleton', [])
    kpt_names = ds_config.get('kpt_names', {})

    default_kpt_names = list(kpt_names.values())[0] if kpt_names else \
        [f'keypoint_{i}' for i in range(num_keypoints)]

    # Use original index-based names for extracted keypoints (consistent with RTMPose config)
    extracted_kpt_names = []
    for i in range(kp_start_idx, kp_end_idx):
        if i < len(default_kpt_names):
            extracted_kpt_names.append(default_kpt_names[i])
        else:
            extracted_kpt_names.append(f'kp_{i}')
    extracted_num_kp = kp_end_idx - kp_start_idx

    # Filter skeleton: keep only edges where both endpoints are in [kp_start_idx, kp_end_idx)
    # and remap to 0-based indices (then back to 1-indexed COCO format)
    filtered_skeleton = []
    for edge in skeleton:
        e0, e1 = edge[0] - 1, edge[1] - 1  # COCO skeleton is 1-indexed
        if kp_start_idx <= e0 < kp_end_idx and kp_start_idx <= e1 < kp_end_idx:
            new_edge = [e0 - kp_start_idx + 1, e1 - kp_start_idx + 1]
            filtered_skeleton.append(new_edge)

    # If filtered skeleton is incomplete, generate a default closed-loop skeleton
    # e.g. for 4 keypoints: [[0,1],[1,2],[2,3],[3,0]] (0-indexed)
    if len(filtered_skeleton) < extracted_num_kp:
        filtered_skeleton = [
            [i, (i + 1) % extracted_num_kp]
            for i in range(extracted_num_kp)
        ]

    if not os.path.isabs(train_file):
        train_file = os.path.join(dataset_dir, train_file)

    with open(train_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    if sub_images_dir is None:
        out_dir = os.path.dirname(out_file)
        sub_images_dir = os.path.join(out_dir, '..', 'images')
    sub_images_dir = os.path.abspath(sub_images_dir)
    os.makedirs(sub_images_dir, exist_ok=True)

    # Triangle sub-images directory (instances whose 4 keypoints form a near-triangle).
    # Default: a sibling directory to sub_images_dir, suffixed with '_triangle'.
    if triangle_out_dir is None:
        triangle_out_dir = sub_images_dir.rstrip('/').rstrip('\\') + '_triangle'
    triangle_out_dir = os.path.abspath(triangle_out_dir)
    os.makedirs(triangle_out_dir, exist_ok=True)

    # Build categories: only classes with id >= min_category_id, remapped to 0..N-1
    kept_class_ids = sorted([cid for cid in range(nc) if cid >= min_category_id])
    cat_id_mapping = {old_id: new_id for new_id, old_id in enumerate(kept_class_ids)}

    categories = []
    for old_id in kept_class_ids:
        cat_name = class_names.get(old_id, f'class_{old_id}')
        new_id = cat_id_mapping[old_id]
        categories.append({
            'id': new_id,
            'name': cat_name,
            'keypoints': extracted_kpt_names,
            'skeleton': filtered_skeleton,
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
    triangle_count = 0
    kpt_count_validated = False

    # Per-category counters: original (from YOLO) and kept (after filtering)
    cat_original_counts = {cid: 0 for cid in range(nc)}
    cat_kept_counts = {new_id: 0 for new_id in cat_id_mapping.values()}

    labels_dir = os.path.join(dataset_dir, 'labels')
    images_dir = os.path.join(dataset_dir, 'images')

    for img_path in image_paths:
        if os.path.isabs(img_path):
            img_abs_path = img_path
        elif os.path.exists(img_path):
            img_abs_path = os.path.abspath(img_path)
        elif os.path.exists(os.path.join(dataset_dir, img_path)):
            img_abs_path = os.path.join(dataset_dir, img_path)
        else:
            img_abs_path = os.path.join(dataset_dir, img_path)

        if not os.path.exists(img_abs_path):
            skipped_no_image += 1
            continue

        image = cv2.imread(img_abs_path)
        if image is None:
            print(f"Warning: Cannot read image {img_abs_path}")
            skipped_no_image += 1
            continue

        img_height, img_width = image.shape[:2]

        rel_path = os.path.relpath(img_abs_path, images_dir)
        label_rel_path = os.path.splitext(rel_path)[0] + '.txt'
        label_abs_path = os.path.join(labels_dir, label_rel_path)

        if not os.path.exists(label_abs_path):
            skipped_no_label += 1
            continue

        has_valid_object = False

        with open(label_abs_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                values = list(map(float, line.split()))
                if len(values) < 5:
                    continue

                class_id = int(values[0])
                if class_id < min_category_id:
                    skipped_no_object += 1
                    continue

                # Validate keypoint count against config
                kpt_values = values[5:]
                actual_num_kpt = len(kpt_values) // 3
                if not kpt_count_validated:
                    if actual_num_kpt != num_keypoints:
                        raise ValueError(
                            f"Keypoint count mismatch! Config says {num_keypoints} keypoints "
                            f"(kpt_shape: {kpt_shape}), but label file has {actual_num_kpt} keypoints "
                            f"({len(kpt_values)} values after bbox).\n"
                            f"  Label file: {label_abs_path}\n"
                            f"  Line: {line[:100]}...\n"
                            f"Please fix dataset.yaml kpt_shape or the label data."
                        )
                    kpt_count_validated = True

                cx, cy, bw, bh = values[1:5]

                # Convert to absolute pixel coordinates
                abs_x = (cx - bw / 2.0) * img_width
                abs_y = (cy - bh / 2.0) * img_height
                abs_w = bw * img_width
                abs_h = bh * img_height

                # Count original category (before any filtering)
                cat_original_counts[class_id] = cat_original_counts.get(class_id, 0) + 1

                # Filter by aspect ratio and short side
                if abs_w > 0 and abs_h > 0:
                    aspect_ratio = max(abs_w, abs_h) / min(abs_w, abs_h)
                    short_side = min(abs_w, abs_h)
                    if aspect_ratio > 3.0 or short_side < 15.0:
                        skipped_shape += 1
                        continue

                # Expand bbox
                crop_x, crop_y, crop_w, crop_h = expand_bbox(
                    abs_x, abs_y, abs_w, abs_h,
                    img_width, img_height, expand_ratio
                )

                if crop_w <= 0 or crop_h <= 0:
                    skipped_no_object += 1
                    continue

                # Crop sub-image
                x1 = int(round(crop_x))
                y1 = int(round(crop_y))
                x2 = int(round(crop_x + crop_w))
                y2 = int(round(crop_y + crop_h))

                sub_image = image[y1:y2, x1:x2]

                # Generate sub-image filename
                img_basename = os.path.splitext(os.path.basename(img_abs_path))[0]
                sub_img_name = f"{img_basename}_{x1}_{y1}_{class_id}.jpg"
                sub_img_path = os.path.join(sub_images_dir, sub_img_name)
                cv2.imwrite(sub_img_path, sub_image)

                # Add image entry for this sub-image
                images.append({
                    'id': image_id,
                    'file_name': sub_img_name,
                    'width': int(crop_w),
                    'height': int(crop_h),
                })

                # Parse keypoints and adjust relative to crop.
                # NOTE: ALL 4 keypoints (indices kp_start_idx..kp_end_idx-1) are
                # treated as "present" for both the triangle test AND the output
                # annotation. Only their visibility flag `v` is normalized so
                # RTMPose does not skip them. Coordinates are preserved as-is.
                keypoints = []
                # Original YOLO coordinates for the triangle test (before crop shift,
                # so v=0 points keep their raw annotation values — important
                # because a v=0 point at (0,0) in the original space expands the
                # hull and prevents false-positives on genuinely degenerate cases).
                orig_pts = []
                for i in range(kp_start_idx, kp_end_idx):
                    base = i * 3
                    if base + 2 < len(kpt_values):
                        kx_orig = kpt_values[base]       # normalized, 0-1
                        ky_orig = kpt_values[base + 1]
                        kv = int(kpt_values[base + 2])
                    else:
                        kx_orig, ky_orig, kv = 0.0, 0.0, 0
                    # Adjust to crop-relative for the annotation output.
                    kx = kx_orig * img_width - crop_x
                    ky = ky_orig * img_height - crop_y
                    # Normalize visibility: v=0 → 1 so RTMPose treats it as visible.
                    out_kv = 1 if kv <= 0 else kv
                    keypoints.extend([kx, ky, out_kv])
                    orig_pts.append((kx_orig, ky_orig))

                # Number of keypoints reported as visible in the output.
                num_visible = sum(
                    1 for i in range(0, len(keypoints), 3) if keypoints[i + 2] > 0
                )

                # ── Triangle-like detection ────────────────────────────────
                # Decision is based on the FULL set of 4 original YOLO coordinates
                # (before crop shift) — visibility flags are ignored. A v=0 point
                # whose raw coordinate is (0,0) or garbage simply expands the hull
                # and makes the ratio smaller, so it will NOT cause false filtering.
                is_tri, _tri_area, _hull_area, _ratio = is_triangle_like(
                    orig_pts, min_ratio=triangle_min_ratio
                )
                if is_tri and len(orig_pts) >= 3:
                    tri_img_path = os.path.join(triangle_out_dir, sub_img_name)
                    if os.path.exists(sub_img_path):
                        os.rename(sub_img_path, tri_img_path)
                    images.pop()  # Drop the image entry we added above.
                    triangle_count += 1
                    # IMPORTANT: do not append to `annotations`, so this instance
                    # does NOT appear in the final COCO JSON output.
                    image_id += 1
                    continue
                # ──────────────────────────────────────────────────────────

                # New bbox: actual object bbox relative to the crop
                obj_x = abs_x - crop_x
                obj_y = abs_y - crop_y
                obj_w = abs_w
                obj_h = abs_h

                annotations.append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': cat_id_mapping[class_id],
                    'bbox': [obj_x, obj_y, obj_w, obj_h],
                    'area': obj_w * obj_h,
                    'keypoints': keypoints,
                    'num_keypoints': num_visible,
                    'iscrowd': 0,
                })

                ann_id += 1
                image_id += 1
                total_objects += 1
                total_cropped += 1
                cat_kept_counts[cat_id_mapping[class_id]] += 1
                has_valid_object = True

    coco_dict = {
        'info': {
            'description': f'Converted from YOLO format: {train_file}',
            'version': '1.0',
            'year': 2026,
        },
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }

    os.makedirs(os.path.dirname(out_file) if os.path.dirname(out_file) else '.', exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(coco_dict, f, indent=2)

    print(f"=" * 60)
    print(f"YOLO to COCO (Single-Object Crop) Summary")
    print(f"=" * 60)
    print(f"Dataset:       {dataset_dir}")
    print(f"Train file:    {train_file}")
    print(f"Output:        {out_file}")
    print(f"Sub-images:    {sub_images_dir}")
    print(f"Triangle dir:  {triangle_out_dir}")
    print(f"-" * 60)
    print(f"Sub-images:    {total_cropped}")
    print(f"Annotations:   {total_objects}")
    print(f"Triangle-like: {triangle_count} (saved separately, excluded from annotations)")
    print(f"Categories:    {len(categories)} (original: {nc}, kept: ids >= {min_category_id})")
    print(f"  Mapping:     {cat_id_mapping}")
    print(f"Keypoints:     {extracted_num_kp} (indices {kp_start_idx}-{kp_end_idx - 1})")
    print(f"Expand ratio:  {expand_ratio}")
    print(f"Triangle ratio threshold: {triangle_min_ratio}")
    print(f"-" * 60)
    print(f"Skipped: {skipped_no_image} (no image), {skipped_no_label} (no label), "
          f"{skipped_no_object} (category < {min_category_id}), "
          f"{skipped_shape} (bad shape), "
          f"{skipped_few_kpt} (< 2 visible kpts; rule disabled — see triangle filter)")
    print(f"-" * 60)
    print(f"Per-category breakdown (original -> kept):")
    for old_id in sorted(kept_class_ids):
        new_id = cat_id_mapping[old_id]
        cat_name = class_names.get(old_id, f'class_{old_id}')
        orig = cat_original_counts.get(old_id, 0)
        kept = cat_kept_counts.get(new_id, 0)
        print(f"  [{old_id:2d} -> {new_id:2d}] {cat_name:20s}: {orig:4d} -> {kept:4d}")
    print(f"=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess COCO annotations')
    parser.add_argument('--mode', type=str, default='yolo2coco', choices=['filter', 'yolo2coco'],
                        help='Operation mode: filter (default) or yolo2coco')

    # filter mode arguments
    parser.add_argument('--ann_file', default=None, help='[filter] Input COCO JSON annotation file')
    parser.add_argument('--min_category_id', type=int, default=5,
                        help='[filter] Minimum category ID to keep (default: 5)')
    parser.add_argument('--kp_start_idx', type=int, default=3,
                        help='[filter] Starting keypoint index to keep, 0-indexed (default: 3)')
    parser.add_argument('--kp_end_idx', type=int, default=7,
                        help='[filter] Ending keypoint index to keep, exclusive (default: 7)')

    # yolo2coco mode arguments
    parser.add_argument('--dataset_dir', default=None,
                        help='[yolo2coco] Root directory of the dataset')
    parser.add_argument('--train_file', default=None,
                        help='[yolo2coco] Path to the train list file')
    parser.add_argument('--dataset_yaml', default=None,
                        help='[yolo2coco] Path to dataset.yaml (default: dataset_dir/dataset.yaml)')
    parser.add_argument('--sub_images_dir', default="data/pose-dataset/BakingRefine/images",
                        help='[yolo2coco] Directory to save cropped sub-images '
                             '(default: images/ directory next to annotations)')
    parser.add_argument('--expand_ratio', type=float, default=1.25,
                        help='[yolo2coco] Bbox expansion ratio (default: 1.25)')
    parser.add_argument('--triangle_out_dir', type=str, default=None,
                        help='[yolo2coco] Directory to save sub-images whose 4 keypoints '
                             'form an approximate triangle (default: <sub_images_dir>_triangle). '
                             'These instances are NOT included in the output COCO JSON.')
    parser.add_argument('--triangle_min_ratio', type=float, default=0.92,
                        help='[yolo2coco] Threshold for triangle-likeness. The largest triangle '
                             'area among the 4 points must be at least this fraction of the '
                             'convex hull area to qualify (default: 0.92).')

    # Common arguments
    parser.add_argument('--out_file', required=True, help='Output COCO JSON file path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'yolo2coco':
        if not args.dataset_dir:
            raise ValueError('--dataset_dir is required for yolo2coco mode')
        if not args.train_file:
            raise ValueError('--train_file is required for yolo2coco mode')
        yolo_to_coco(
            dataset_dir=args.dataset_dir,
            train_file=args.train_file,
            out_file=args.out_file,
            dataset_yaml=args.dataset_yaml,
            sub_images_dir=args.sub_images_dir,
            expand_ratio=args.expand_ratio,
            min_category_id=args.min_category_id,
            kp_start_idx=args.kp_start_idx,
            kp_end_idx=args.kp_end_idx,  # kp_end_idx in yolo2coco is exclusive, CLI uses inclusive
            triangle_out_dir=args.triangle_out_dir,
            triangle_min_ratio=args.triangle_min_ratio,
        )
    else:
        if not args.ann_file:
            raise ValueError('--ann_file is required for filter mode')
        filter_coco_annotations(
            ann_file=args.ann_file,
            out_file=args.out_file,
            min_category_id=args.min_category_id,
            kp_start_idx=args.kp_start_idx,
            kp_end_idx=args.kp_end_idx,
        )