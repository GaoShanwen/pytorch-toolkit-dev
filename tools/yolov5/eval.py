import os
import cv2
import numpy as np
from tqdm import tqdm
import sys

def iou(boxes1, boxes2, format='xyxy'):
    if format == 'xywh':
        boxes1[:, 2:4] = boxes1[:, 2:4] + boxes1[:, :2]
        boxes2[:, 2:4] = boxes2[:, 2:4] + boxes2[:, :2]
    
    boxes1 = np.expand_dims(boxes1, axis=1)  # 形状变为 (N, 1, 4)
    boxes2 = np.expand_dims(boxes2, axis=0)  # 形状变为 (1, M, 4)
    x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    return intersection / (area1 + area2 - intersection)


def compute_pr(eval_infos, th=0.25, iou_th=0.75):
    tp, fp, fn, gn, pn = 0, 0, 0, 0, 0
    for eval_info in tqdm(eval_infos):
        preds, labels = eval_info.get("pred",[]), eval_info.get("label",[])
        if th is not None:
            preds = [p for p in preds if p[5] >= th]
        txt_path = eval_info.get("txt_path",[])
        if not len(labels):
            fp += len(preds)
            pn += len(preds)
            assert tp+fp==pn and tp+fn==gn, f"{txt_path} compute error!"
            continue
        if not len(preds):
            fn += len(labels)
            gn += len(labels)
            assert tp+fp==pn and tp+fn==gn, f"{txt_path} compute error!"
            continue
        img_size = eval_info["img_size"] * 2
        preds, labels = np.array(preds), np.array(labels)
        gt_boxes = labels[:, 1:5] * img_size
        dt_boxes = preds[:, 1:5] * img_size
        for cat_id in set(labels[:,0].tolist()+preds[:,0].tolist()):
            gts = gt_boxes[labels[:, 0]==cat_id]
            pts = dt_boxes[preds[:, 0]==cat_id]
            gn += len(gts)
            pn += len(pts)
            if not len(gts):
                fp += len(pts)
                continue
            if not len(pts):
                fn += len(gts)
                continue
            iou_matrix = iou(gts, pts, "xywh")
            tp += (np.max(iou_matrix, axis=1)>iou_th).sum()
            fp += (np.max(iou_matrix, axis=0)<=iou_th).sum()
            fn += (np.max(iou_matrix, axis=1)<=iou_th).sum()
        assert tp+fp==pn and tp+fn==gn, f"{txt_path} compute error!"

    precision, recall = tp / pn, (gn-fn) / gn 
    print(f"(th={th})-TP/FP/FN: {tp}/{fp}/{fn};preds/gts: {pn}/{gn}")
    return precision, recall


def load_preds(pred_dir, anno_dir, img_dir):
    assert os.path.exists(pred_dir) and os.path.exists(anno_dir), "please make sure pred and label dir is exist"
    file_list = [file_name for file_name in os.listdir(img_dir) if file_name.endswith(".jpg")]
    eval_infos = []
    for img_name in tqdm(file_list):
        img_path = os.path.join(img_dir, img_name)
        h, w, _ = cv2.imread(img_path).shape
        label_name = img_name.replace(".jpg", ".txt")
        pred_path = os.path.join(pred_dir, label_name)
        pts = []
        if os.path.exists(pred_path):
            with open(pred_path, 'r') as f:
                pts = [list(map(float, line.strip().split())) for line in f.readlines()]

        gt_path = os.path.join(anno_dir, label_name)
        gts = []
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                gts = [list(map(float, line.strip().split())) for line in f.readlines()]
        eval_infos.append({"txt_path": gt_path, "img_size": [w, h], "pred":pts, "label": gts})
    return eval_infos


if __name__=="__main__":
    pred_dir = sys.argv[1] #"data/pred_labels"
    gt_dir = sys.argv[2] #"data/labels"
    img_dir = sys.argv[3] #"data/images"
    eval_infos = load_preds(pred_dir, gt_dir, img_dir)

    p, r = compute_pr(eval_infos, th=0.0)
    print(f"precision={p*100:.2f}/recall={r*100:.2f}")
