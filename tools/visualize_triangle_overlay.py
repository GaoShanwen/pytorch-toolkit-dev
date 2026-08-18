"""
Direct verification: re-read YOLO labels for the saved triangle instances,
overlay their 4 keypoints, and visually confirm they form near-triangles.

This walks the actual YOLO labels (not just file-name parsing), so it should
work for any source path. We identify each saved triangle image by:
    (1) class id in filename, (2) matching x/y crop offset against the bbox,
    (3) the convex-hull ratio threshold.
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
    p.add_argument("--dataset_dir", default="/home/wenjie/workspace/pytorch-toolkit-dev/data/pose-dataset/BakingRecognize")
    p.add_argument("--ann", default="/home/wenjie/workspace/pytorch-toolkit-dev/data/pose-dataset/BakingRefine/annotations/val_v3.json")
    p.add_argument("--tri_dir", default="/home/wenjie/workspace/pytorch-toolkit-dev/data/pose-dataset/BakingRefine/images_v3_triangle")
    p.add_argument("--out", default="/home/wenjie/workspace/pytorch-toolkit-dev/runs/visualize_triangle_v3/03_triangle_kps_overlay.jpg")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def hull_area(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return 0.0
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo = []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    up = []
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    hull = lo[:-1] + up[:-1]
    if len(hull) < 3:
        return 0.0
    area = 0.0
    n = len(hull)
    for i in range(n):
        j = (i+1) % n
        area += hull[i][0]*hull[j][1] - hull[j][0]*hull[i][1]
    return abs(area) * 0.5


def main():
    args = parse_args()
    random.seed(args.seed)

    coco = json.load(open(args.ann))
    cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}

    # Build a quick lookup: (basename, class_id) -> list of label rows.
    # We need to match by also considering the (x0, y0) crop offset in the filename.
    pat = re.compile(r"^(.+)_(\d+)_(\d+)_(\d+)\.jpg$")

    samples = []
    for fn in os.listdir(args.tri_dir):
        m = pat.match(fn)
        if not m:
            continue
        basename, x0, y0, cls_old = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        # Find original source image and label
        img_dir = os.path.join(args.dataset_dir, "images")
        # Try direct, then recursively
        candidates = []
        for ext in (".png", ".jpg", ".jpeg"):
            cand = os.path.join(img_dir, basename + ext)
            if os.path.exists(cand):
                candidates = [cand]; break
        if not candidates:
            import glob
            found = glob.glob(os.path.join(img_dir, "**", basename + ".*"), recursive=True)
            for f in found:
                if f.endswith((".png", ".jpg", ".jpeg")):
                    candidates = [f]; break
        if not candidates:
            continue
        src_path = candidates[0]
        src = cv2.imread(src_path)
        if src is None:
            continue
        H, W = src.shape[:2]
        rel = os.path.relpath(src_path, img_dir)
        lab = os.path.join(args.dataset_dir, "labels",
                           os.path.splitext(rel)[0] + ".txt")
        if not os.path.exists(lab):
            continue
        # Find line with the right class id and matching (x0, y0)
        with open(lab) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        for line in lines:
            vals = list(map(float, line.split()))
            if not vals or int(vals[0]) != cls_old:
                continue
            cx, cy, bw, bh = vals[1], vals[2], vals[3], vals[4]
            abs_x = (cx - bw/2.0) * W
            abs_y = (cy - bh/2.0) * H
            abs_w = bw * W
            abs_h = bh * H
            # Expand ratio = 1.25
            new_w = abs_w * 1.25
            new_h = abs_h * 1.25
            ex0 = max(0.0, abs_x + abs_w/2.0 - new_w/2.0)
            ey0 = max(0.0, abs_y + abs_h/2.0 - new_h/2.0)
            # round to int (the script uses int(round(crop_x)))
            if int(round(ex0)) != x0 or int(round(ey0)) != y0:
                continue
            # extract visible keypoints 4..7
            kpts = []
            for i in range(4, 8):
                base = i * 3
                if 5 + base + 2 < len(vals):
                    kx = vals[5 + base] * W - ex0
                    ky = vals[5 + base + 1] * H - ey0
                    kv = int(vals[5 + base + 2])
                    kpts.append((kx, ky, kv))
                else:
                    kpts.append((0.0, 0.0, 0))
            # Compute ratio
            vis = [(kx, ky) for kx, ky, kv in kpts if kv > 0]
            if len(vis) >= 3:
                ha = hull_area(vis)
                if ha > 0:
                    best = 0.0
                    n = len(vis)
                    for i in range(n-2):
                        for j in range(i+1, n-1):
                            for k in range(j+1, n):
                                ax, ay = vis[i]; bx, by = vis[j]; cx_, cy_ = vis[k]
                                a = abs((bx-ax)*(cy_-ay) - (by-ay)*(cx_-ax)) * 0.5
                                if a > best: best = a
                    ratio = best / ha
                else:
                    ratio = 0
            else:
                ratio = 0
            samples.append({
                "fn": fn, "kpts": kpts, "ratio": ratio, "cls_old": cls_old,
                "src": src_path, "x0": x0, "y0": y0,
            })
            break

    # Sort by ratio ascending (most triangle-like first)
    samples.sort(key=lambda s: -s["ratio"])
    pick = samples[:args.n]

    out_imgs = []
    labels = []
    for s in pick:
        im = cv2.imread(os.path.join(args.tri_dir, s["fn"]))
        if im is None:
            continue
        h, w = im.shape[:2]
        # Draw keypoints in crop coords
        pts = []
        for kx, ky, kv in s["kpts"]:
            if kv <= 0:
                continue
            cx, cy = int(round(kx)), int(round(ky))
            pts.append((cx, cy))
            cv2.circle(im, (cx, cy), max(3, w // 35), (0, 0, 255), -1)
        # Convex hull (filled, semi-transparent)
        if len(pts) >= 3:
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            overlay = im.copy()
            cv2.fillConvexPoly(overlay, hull, (255, 200, 0))
            cv2.addWeighted(overlay, 0.25, im, 0.75, 0, im)
            cv2.polylines(im, [hull], True, (0, 255, 255), 2)
            # connect visible points sequentially
            for i in range(len(pts) - 1):
                cv2.line(im, pts[i], pts[i+1], (0, 255, 0), 2)
        cat_name = cat_id_to_name.get(s["cls_old"] - 5, "?") \
                   if (s["cls_old"] - 5) in cat_id_to_name else f"cls{s['cls_old']}"
        cv2.rectangle(im, (0, 0), (w, h // 16), (0, 0, 0), -1)
        cv2.putText(im, f"ratio={s['ratio']:.3f} cls={cat_name}",
                    (4, h // 16 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        out_imgs.append(im)
        labels.append(s["fn"][:48])

    # Tile
    if not out_imgs:
        print("No samples recovered")
        return
    tile_h = max(im.shape[0] for im in out_imgs)
    tile_w = max(im.shape[1] for im in out_imgs)
    cols = 4
    rows = (len(out_imgs) + cols - 1) // cols
    grid = np.full((rows * tile_h, cols * tile_w, 3), 32, dtype=np.uint8)
    for idx, im in enumerate(out_imgs):
        r, c = divmod(idx, cols)
        canvas = np.full((tile_h, tile_w, 3), 0, dtype=np.uint8)
        th, tw = im.shape[:2]
        canvas[:th, :tw] = im
        grid[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = canvas
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"saved {args.out} with {len(out_imgs)} samples")
    print(f"ratio range: [{min(s['ratio'] for s in pick):.3f}, "
          f"{max(s['ratio'] for s in pick):.3f}]")


if __name__ == "__main__":
    main()