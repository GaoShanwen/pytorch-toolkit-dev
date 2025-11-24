import os
import cv2
import numpy as np
from tqdm import tqdm
import yaml
import json
import torch
import sys
sys.path.append('.')
from vis_box import make_parser, generate_colors
sys.path.append('...')
from local_lib.data.utils import box_ioa

def iou(boxes1, boxes2, format='xyxy'):
    if format == 'xywh':
        boxes1[:, 2:4] = boxes1[:, 2:4] + boxes1[:, :2]
        boxes2[:, 2:4] = boxes2[:, 2:4] + boxes2[:, :2]
    # 扩展维度以便进行广播计算
    boxes1 = np.expand_dims(boxes1, axis=1)  # 形状变为 (N, 1, 4)
    boxes2 = np.expand_dims(boxes2, axis=0)  # 形状变为 (1, M, 4)
    x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou

def vis_img(img_path, preds, labels, colors, fp, fn, tp, obj_dir):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    def do_vis(_img, boxes):
        for box in boxes:
            l, x, y, w, h, *_ = box
            if l > len(categories):
                continue
            l = int(l)
            color = colors[l]

            w, h = round(width * w), round(height * h)
            x, y = round(width * x - w / 2), round(height * y - h / 2)
            name = categories[l] + str(*_)
            cv2.putText(_img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(_img, (x, y), (x + w, y + h), color, 2)
        return _img
    blank_image = np.zeros((height, width*2, 3), dtype=np.uint8)
    blank_image[:, :width, :] = do_vis(img.copy(), labels)
    blank_image[:, width:, :] = do_vis(img, preds)
    cv2.putText(blank_image, f"tp: {tp}; fn: {fn}", (20, height-100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, colors[-1], 4)
    cv2.putText(blank_image, f"fp: {fp}", (width+20, height-100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, colors[-1], 4)
    folder_name = "fp" if not fn and fp else "fn" if not fp and fn else "other" 
    cv2.imwrite(os.path.join(obj_dir, folder_name, os.path.basename(img_path)), blank_image)

def compute_pr(eval_infos, th=0.25, iou_th=0.75, cats=[], obj_root="", ioav=0.25, do_vis=True):
    tp, fp, fn, gn, pn = 0, 0, 0, 0, 0
    last_fp, last_fn, last_tp = 0, 0, 0
    colors = generate_colors(len(cats))
    for eval_info in tqdm(eval_infos):
        if do_vis and fp > last_fp or fn > last_fn:
            e_p, e_g = np.vstack(e_p) if len(e_p) else [], np.vstack(e_g) if len(e_g) else []
            vis_img(img_path, e_p, e_g, colors, fp-last_fp, fn-last_fn, tp-last_tp, obj_root)
        last_fp, last_fn, last_tp, img_path = fp, fn, tp, eval_info.get("img_path", None)
        e_p, e_g = [], []
        preds, labels = eval_info.get("pred",[]), eval_info.get("label",[])
        labels = np.array(labels)
        img_size = eval_info["img_size"] * 2
        if len(labels):
            try:
                gt_boxes = labels[:, 1:5] * img_size
                ignore_idx = labels[:, -1].astype(bool) # iscrowd
                labels = labels[~ignore_idx, :-1]
            except Exception as e:
                print(eval_info)
                labels = []
        else:
            ignore_idx = np.array([], dtype=bool)
        if th is not None:
            preds = [p for p in preds if p[5] >= th]
        preds = np.array(preds)
        if len(preds):
            dt_boxes = preds[:, 1:5] * img_size
            if ignore_idx.sum():
                pred_keeps = (
                    box_ioa(
                        torch.from_numpy(dt_boxes).T, torch.from_numpy(gt_boxes[ignore_idx]), 
                        x1y1x2y2=False
                    ) <= ioav
                ).all(dim=0).numpy()
                preds, dt_boxes = preds[pred_keeps], dt_boxes[pred_keeps]
        if not len(labels):
            fp += len(preds)
            pn += len(preds)
            e_p.append(preds)
            continue
        if not len(preds):
            fn += len(labels)
            gn += len(labels)
            e_g.append(labels)
            continue
        dt_boxes = preds[:, 1:5] * img_size
        gt_boxes = gt_boxes[~ignore_idx]
        e_p, e_g = [], []
        for cat_id in set(labels[:,0].tolist()):
            gt_keeps = labels[:, 0] == cat_id
            gts = gt_boxes[gt_keeps]
            pt_keeps = preds[:, 0] == cat_id
            try:
                pts = dt_boxes[pt_keeps]
            except:
                print(dt_boxes.tolist(), pt_keeps.tolist())
            if not len(gts):
                fp += len(pts)
                pn += len(pts)
                e_p.append(preds[pt_keeps])
                continue
            if not len(pts):
                fn += len(gts)
                gn += len(gts)
                e_g.append(labels[gt_keeps])
                continue
            iou_matrix = iou(gts, pts, "xywh")
            this_tp = (np.max(iou_matrix, axis=1) > iou_th).sum()
            fp_idx = np.max(iou_matrix, axis=0) <= iou_th
            this_fp = fp_idx.sum()
            fn_idx = np.max(iou_matrix, axis=1) <= iou_th
            this_fn = fn_idx.sum()
            e_p.append(preds[pt_keeps][fp_idx])
            e_g.append(labels[gt_keeps][fn_idx])

            tp += this_tp
            fp += this_fp
            fn += this_fn
        gn += len(gt_boxes)
        pn += len(dt_boxes)

    precision, recall = tp / max(1,pn), (gn-fn) / max(1,gn)
    print(f"(th={th})-TP/FP/FN:{tp}/{fp}/{fn}")
    return precision, recall


def load_preds(pred_dir, img_files):
    with open(img_files, 'r') as f:
        imgs = [line.strip() for line in f.readlines()]
    eval_infos = []
    for img_path in tqdm(imgs):
        try:
            gts = []
            gt_path = img_path.replace(".jpg", ".txt")
            if os.path.exists(gt_path):
                with open(gt_path, 'r') as f:
                    gts = [list(map(float, line.strip().split())) for line in f.readlines()]

            pts = []
            pred_path = os.path.join(pred_dir, os.path.basename(gt_path))
            if os.path.exists(pred_path):
                h, w, _ = cv2.imread(img_path).shape
                with open(pred_path, 'r') as f:
                    pts = [list(map(float, line.strip().split())) for line in f.readlines()]
            eval_infos.append({"img_path": img_path, "img_size": [w, h], "pred": pts, "label": gts})
        except Exception as e:
            print(f"{img_path} error {e}")
    return eval_infos


if __name__=="__main__":
    pred_dir = "runs/detect/val5/labels"
    args = make_parser()
    with open(os.path.join('/'.join(args.src_files.split("/")[:2]), args.task, "ultralytics.yaml"), 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    categories = [v for _, v in data["names"].items()]
    filename = os.path.join('/'.join(pred_dir.split('/')[:-1]), "eval_infos.json")
    eval_infos = load_preds(pred_dir, args.src_files)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(eval_infos, f)
    # with open(filename, 'r', encoding='utf-8') as f:
    #     eval_infos = json.load(f)
    
    obj_dir = os.path.join(args.obj_root, args.task)
    def make_dir(obj_dir):
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)
    for name in ['', 'fp', 'fn', 'other']:
        make_dir(os.path.join(obj_dir, name))
    p, r = compute_pr(eval_infos, th=0.25, iou_th=0.5, cats=categories, obj_root=obj_dir, do_vis=True)
    print(f"precision={p*100:.2f}/recall={r*100:.2f}")
