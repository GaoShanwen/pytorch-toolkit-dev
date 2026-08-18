"""
Visualize preprocessed YOLO→COCO keypoints for both the kept (normal) and
the triangle-filtered sub-images. Saves side-by-side grids so we can eyeball
whether the triangle filter actually rejected only near-triangle instances.

Usage:
    python tools/visualize_triangle_filter.py \
        --ann  data/pose-dataset/BakingRefine/annotations/val_v3.json \
        --keep_dir data/pose-dataset/BakingRefine/images_v3 \
        --tri_dir   data/pose-dataset/BakingRefine/images_v3_triangle \
        --out_dir   runs/visualize_triangle \
        --keep_n 8 --tri_n 8
"""
import argparse
import json
import os
import random

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True)
    p.add_argument("--keep_dir", required=True)
    p.add_argument("--tri_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--keep_n", type=int, default=8)
    p.add_argument("--tri_n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


COLORS = {
    0: (220, 20, 60),    # Tray
    1: (0, 139, 139),    # Tray_Invalid
    2: (255, 140, 0),    # Tabletop
    3: (148, 0, 211),    # Oven_TopHandle
    4: (0, 100, 0),      # Oven_BottomHandle
    5: (70, 130, 180),   # Oven_TopInner
    6: (220, 20, 60),    # Oven_BottomInner
    7: (178, 34, 34),    # Grill
    8: (34, 139, 34),    # Screen_Number
    9: (255, 165, 0),    # Screen_Fuction
}


def draw_keypoints(img, keypoints, cat_id, kp_names):
    out = img.copy()
    h, w = out.shape[:2]
    pts_xy = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
        if v <= 0:
            continue
        cx, cy = int(round(x)), int(round(y))
        pts_xy.append((cx, cy, v))
        cv2.circle(out, (cx, cy), max(3, w // 40), COLORS.get(cat_id, (0, 255, 255)), -1)
        cv2.circle(out, (cx, cy), max(3, w // 40), (0, 0, 0), 1)
    # connect consecutive points to expose the shape
    for i in range(len(pts_xy) - 1):
        cv2.line(out, (pts_xy[i][0], pts_xy[i][1]),
                 (pts_xy[i+1][0], pts_xy[i+1][1]),
                 COLORS.get(cat_id, (0, 255, 255)), 2)
    if len(pts_xy) >= 3:
        cv2.line(out, (pts_xy[-1][0], pts_xy[-1][1]),
                 (pts_xy[0][0], pts_xy[0][1]),
                 COLORS.get(cat_id, (0, 255, 255)), 2)
    return out, pts_xy


def annotate_with_label(img, text):
    out = img.copy()
    h, w = out.shape[:2]
    pad = max(2, h // 60)
    cv2.rectangle(out, (0, 0), (w, h // 18), (0, 0, 0), -1)
    cv2.putText(out, text, (pad, h // 18 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def tile(images, labels, cols, tile_h=None, tile_w=None):
    if not images:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    if tile_h is None or tile_w is None:
        tile_h = max(im.shape[0] for im in images)
        tile_w = max(im.shape[1] for im in images)
    rows = (len(images) + cols - 1) // cols
    grid = np.full((rows * tile_h, cols * tile_w, 3), 32, dtype=np.uint8)
    for idx, (im, label) in enumerate(zip(images, labels)):
        r, c = divmod(idx, cols)
        annotated = annotate_with_label(
            cv2.resize(im, (tile_w, tile_h)), label
        )
        grid[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = annotated
    return grid


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    coco = json.load(open(args.ann))
    cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
    kp_names = coco['categories'][0]['keypoints']

    # Normal / kept images: iterate annotations
    keep_samples = []
    for ann in coco['annotations']:
        img = next(i for i in coco['images'] if i['id'] == ann['image_id'])
        keep_samples.append((img, ann))

    # Triangle images: not in JSON. List directory, but we need to recover the
    # keypoints. The YOLO label still contains them; but easier: re-load from
    # the file name pattern and re-derive visibility.
    tri_files = sorted(os.listdir(args.tri_dir))
    # We don't have keypoint coords for the triangle set in the JSON because
    # they were excluded. Show their sub-images only (no overlays possible).
    # We *can* still derive bbox from image dims.

    random.shuffle(keep_samples)
    random.shuffle(tri_files)

    # ── Tile 1: normal (kept) — with keypoint overlay ──
    kept_imgs, kept_labels = [], []
    for img_meta, ann in keep_samples[:args.keep_n]:
        path = os.path.join(args.keep_dir, img_meta['file_name'])
        if not os.path.exists(path):
            continue
        im = cv2.imread(path)
        if im is None:
            continue
        cat_name = cat_id_to_name.get(ann['category_id'], '?')
        out, _ = draw_keypoints(im, ann['keypoints'], ann['category_id'], kp_names)
        kept_imgs.append(out)
        kept_labels.append(f"{cat_name} | {img_meta['file_name'][:36]}")
    grid_kept = tile(kept_imgs, kept_labels, cols=4)
    cv2.imwrite(os.path.join(args.out_dir, '01_kept_normal.jpg'), grid_kept)
    print(f"saved: {os.path.join(args.out_dir, '01_kept_normal.jpg')}  "
          f"({len(kept_imgs)} samples)")

    # ── Tile 2: triangle-filtered — sub-images without overlays (no kp in JSON) ──
    tri_imgs, tri_labels = [], []
    for fn in tri_files[:args.tri_n]:
        path = os.path.join(args.tri_dir, fn)
        im = cv2.imread(path)
        if im is None:
            continue
        tri_imgs.append(im)
        tri_labels.append(f"{fn[:48]}")
    grid_tri = tile(tri_imgs, tri_labels, cols=4)
    cv2.imwrite(os.path.join(args.out_dir, '02_filtered_triangle.jpg'), grid_tri)
    print(f"saved: {os.path.join(args.out_dir, '02_filtered_triangle.jpg')}  "
          f"({len(tri_imgs)} samples)")

    # ── Tile 3 (verification): re-load source YOLO labels for the triangle
    # set and overlay the 4 keypoints so we can SEE whether they truly form
    # a triangle. This requires knowing the source paths.
    # Use the convention: image_id was incremented per instance, so we can
    # look up the original YOLO line by parsing the file name pattern
    # `<img_basename>_<x>_<y>_<class_id>.jpg`.
    import re
    pattern = re.compile(r"^(.+)_(\d+)_(\d+)_(\d+)\.jpg$")
    dataset_yaml_root = "/home/wenjie/workspace/pytorch-toolkit-dev/data/pose-dataset/BakingRecognize"

    def recover_kpts(filename):
        m = pattern.match(filename)
        if not m:
            return None, None, None, None
        basename, x0, y0, cls_old = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        # The original image is the basename + extension.
        candidates = []
        for ext in (".png", ".jpg", ".jpeg"):
            for sub in (
                f"images/{basename}{ext}",
                f"images/{basename[:30]}/{basename}{ext}",
            ):
                p = os.path.join(dataset_yaml_root, sub)
                if os.path.exists(p):
                    candidates.append(p)
                    break
            if candidates:
                break
        # Fall back to glob
        if not candidates:
            import glob as _g
            found = _g.glob(os.path.join(dataset_yaml_root, "**", basename + ".*"), recursive=True)
            for f in found:
                if f.endswith((".png", ".jpg", ".jpeg")):
                    candidates = [f]
                    break
        if not candidates:
            return None, None, None, None
        src_path = candidates[0]
        rel = os.path.relpath(src_path, os.path.join(dataset_yaml_root, "images"))
        label_path = os.path.join(dataset_yaml_root, "labels", os.path.splitext(rel)[0] + ".txt")
        if not os.path.exists(label_path):
            return None, None, None, None
        # Find the right line (matching class_id)
        with open(label_path) as f:
            for line in f:
                vals = list(map(float, line.split()))
                if not vals:
                    continue
                if int(vals[0]) != cls_old:
                    continue
                # crop to (kp_start_idx=4..7 inclusive) — indices 4,5,6,7
                kpts = []
                for i in range(4, 8):
                    base = i * 3
                    kx = vals[5 + base] if base + 2 < len(vals) - 5 else 0.0
                    ky = vals[5 + base + 1] if base + 2 < len(vals) - 5 else 0.0
                    kv = int(vals[5 + base + 2]) if base + 2 < len(vals) - 5 else 0
                    kpts.append((kx, ky, kv))
                return src_path, kpts, (x0, y0), cls_old
        return None, None, None, None

    tri_imgs2, tri_labels2 = [], []
    for fn in tri_files[:args.tri_n]:
        src, kpts, origin, cls_old = recover_kpts(fn)
        if src is None:
            continue
        im = cv2.imread(os.path.join(args.tri_dir, fn))
        if im is None:
            continue
        x0, y0 = origin
        # Draw keypoints in crop coords
        h, w = im.shape[:2]
        pts = []
        for kx_n, ky_n, kv in kpts:
            if kv <= 0:
                continue
            cx = int(round(kx_n * cv2.imread(src).shape[1] - x0))
            cy = int(round(ky_n * cv2.imread(src).shape[0] - y0))
            pts.append((cx, cy))
            cv2.circle(im, (cx, cy), max(3, w // 40), (0, 0, 255), -1)
        # connect to highlight shape
        for i in range(len(pts) - 1):
            cv2.line(im, pts[i], pts[i+1], (0, 255, 0), 2)
        if len(pts) >= 3:
            cv2.line(im, pts[-1], pts[0], (0, 255, 0), 2)
        tri_imgs2.append(im)
        cat_name = cat_id_to_name.get(cls_old - 5, "?") if cls_old - 5 in cat_id_to_name else f"cls{cls_old}"
        tri_labels2.append(f"old_cls={cls_old} {cat_name} | {fn[:30]}")
    grid_tri2 = tile(tri_imgs2, tri_labels2, cols=4)
    cv2.imwrite(os.path.join(args.out_dir, '03_triangle_kps_overlay.jpg'), grid_tri2)
    print(f"saved: {os.path.join(args.out_dir, '03_triangle_kps_overlay.jpg')}  "
          f"({len(tri_imgs2)} samples)")


if __name__ == '__main__':
    main()