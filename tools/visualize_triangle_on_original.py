"""
Draw keypoints for ALL triangle instances directly on the ORIGINAL images
(no cropping). Each row = one source image, multiple boxes = multiple
triangle-classified instances inside it.

Output: runs/visualize_triangle_v3/original_with_kps.jpg
"""
import argparse
import json
import os
import random
import re

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir",
                   default="/home/wenjie/workspace/pytorch-toolkit-dev/data/pose-dataset/BakingRecognize")
    p.add_argument("--tri_dir",
                   default="/home/wenjie/workspace/pytorch-toolkit-dev/data/pose-dataset/BakingRefine/images_v3_triangle")
    p.add_argument("--out",
                   default="/home/wenjie/workspace/pytorch-toolkit-dev/runs/visualize_triangle_v3/original_with_kps.jpg")
    p.add_argument("--max_imgs", type=int, default=8,
                   help="max source images to draw")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# Stable colors per original class id (5..14 in this dataset).
COLORS = {
    5: (220, 20, 60),
    6: (0, 139, 139),
    7: (255, 140, 0),
    8: (148, 0, 211),
    9: (0, 100, 0),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (178, 34, 34),
    13: (34, 139, 34),
    14: (255, 165, 0),
}
CLASS_NAMES = {
    5: "Tray", 6: "Tray_Invalid", 7: "Tabletop", 8: "Oven_TopHandle",
    9: "Oven_BottomHandle", 10: "Oven_TopInner", 11: "Oven_BottomInner",
    12: "Grill", 13: "Screen_Number", 14: "Screen_Fuction",
}


def main():
    args = parse_args()
    random.seed(args.seed)

    # Group triangle files by their original image basename.
    pat = re.compile(r"^(.+)_(\d+)_(\d+)_(\d+)\.jpg$")
    by_src = {}  # basename -> list of (fn, x0, y0, cls_old)
    for fn in os.listdir(args.tri_dir):
        m = pat.match(fn)
        if not m:
            continue
        basename, x0, y0, cls_old = (m.group(1), int(m.group(2)),
                                       int(m.group(3)), int(m.group(4)))
        by_src.setdefault(basename, []).append((fn, x0, y0, cls_old))

    # Find original images. Group may live in a sub-directory.
    img_dir = os.path.join(args.dataset_dir, "images")
    candidates = {}
    for basename in by_src:
        for ext in (".png", ".jpg", ".jpeg"):
            cand = os.path.join(img_dir, basename + ext)
            if os.path.exists(cand):
                candidates[basename] = cand
                break
        if basename not in candidates:
            import glob
            found = glob.glob(os.path.join(img_dir, "**", basename + ".*"), recursive=True)
            for f in found:
                if f.endswith((".png", ".jpg", ".jpeg")):
                    candidates[basename] = f
                    break

    # Read the matching YOLO label files for those images.
    label_data = {}
    for basename, src_path in candidates.items():
        rel = os.path.relpath(src_path, img_dir)
        lab = os.path.join(args.dataset_dir, "labels", os.path.splitext(rel)[0] + ".txt")
        if not os.path.exists(lab):
            continue
        with open(lab) as f:
            rows = []
            for line in f.read().splitlines():
                vals = list(map(float, line.split()))
                if not vals:
                    continue
                rows.append(vals)
        label_data[basename] = (src_path, rows)

    # Pick images with multiple triangle instances first, otherwise just any.
    keys = sorted(by_src.keys(),
                  key=lambda k: -len(by_src[k]))
    keys = [k for k in keys if k in label_data][:args.max_imgs]

    # Build the per-source visualization.
    panels = []
    for basename in keys:
        src_path, rows = label_data[basename]
        src = cv2.imread(src_path)
        if src is None:
            continue
        H, W = src.shape[:2]

        # For each triangle instance, draw keypoints and bbox on the source.
        for fn, x0, y0, cls_old in by_src[basename]:
            color = COLORS.get(cls_old, (0, 255, 255))
            # Find the matching label row by (class id, crop offset).
            for vals in rows:
                if int(vals[0]) != cls_old:
                    continue
                cx_, cy_, bw_, bh_ = vals[1], vals[2], vals[3], vals[4]
                abs_x = (cx_ - bw_/2.0) * W
                abs_y = (cy_ - bh_/2.0) * H
                abs_w = bw_ * W
                abs_h = bh_ * H
                new_w = abs_w * 1.25
                new_h = abs_h * 1.25
                ex0 = max(0.0, abs_x + abs_w/2.0 - new_w/2.0)
                ey0 = max(0.0, abs_y + abs_h/2.0 - new_h/2.0)
                if int(round(ex0)) != x0 or int(round(ey0)) != y0:
                    continue

                # bbox on source image
                bx1, by1 = int(round(abs_x)), int(round(abs_y))
                bx2, by2 = int(round(abs_x + abs_w)), int(round(abs_y + abs_h))
                cv2.rectangle(src, (bx1, by1), (bx2, by2), color, 2)

                # 4 keypoints (indices 4..7) in source coords
                pts = []
                for i in range(4, 8):
                    base = i * 3
                    if 5 + base + 2 < len(vals):
                        kx = vals[5 + base] * W
                        ky = vals[5 + base + 1] * H
                        kv = int(vals[5 + base + 2])
                        if kv > 0:
                            pts.append((int(round(kx)), int(round(ky))))
                # draw convex hull + edges
                if len(pts) >= 3:
                    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                    overlay = src.copy()
                    cv2.fillConvexPoly(overlay, hull, color)
                    cv2.addWeighted(overlay, 0.20, src, 0.80, 0, src)
                    cv2.polylines(src, [hull], True, color, 2)
                    for i in range(len(pts) - 1):
                        cv2.line(src, pts[i], pts[i+1], (0, 255, 0), 2)
                    if len(pts) >= 3:
                        cv2.line(src, pts[-1], pts[0], (0, 255, 0), 2)
                for p in pts:
                    cv2.circle(src, p, 6, (0, 0, 255), -1)
                    cv2.circle(src, p, 6, (0, 255, 255), 2)

                # label box top-left
                name = CLASS_NAMES.get(cls_old, f"cls{cls_old}")
                cv2.rectangle(src, (bx1, by1 - 22), (bx1 + 130, by1), color, -1)
                cv2.putText(src, name, (bx1 + 4, by1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                break

        # Top header: which source image + count
        n_tri = len(by_src[basename])
        cv2.rectangle(src, (0, 0), (W, 36), (0, 0, 0), -1)
        cv2.putText(src,
                    f"{os.path.basename(src_path)}  |  {n_tri} triangle inst.",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(src)

    if not panels:
        print("nothing to draw")
        return

    # Tile vertically with a 4-col layout to keep images legible.
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    tile_h = max(p.shape[0] for p in panels)
    tile_w = max(p.shape[1] for p in panels)
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 32, dtype=np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        th, tw = p.shape[:2]
        # center inside the tile
        y0 = r*tile_h + (tile_h - th) // 2
        x0 = c*tile_w + (tile_w - tw) // 2
        canvas[y0:y0+th, x0:x0+tw] = p

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, canvas)
    print(f"saved {args.out}  ({len(panels)} source images)")


if __name__ == "__main__":
    main()