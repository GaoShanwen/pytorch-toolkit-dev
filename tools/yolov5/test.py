import os
import json
import torch
import pickle
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.metrics import ap_per_class
from utils.general import box_iou


def xywh2xyxy(boxes):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
    return boxes


def load_preds(file_dir, anno_path, task="JY-M"):
    suffix = "" if task == "V5M1280" else ".pkl"
    pred_path = os.path.join(file_dir, task+suffix)
    preds = []
    if task.startswith("CoDCN"):
        with open(pred_path, 'rb') as file:
            data = pickle.load(file)
        for i, res in enumerate(data):
            for j, boxes in enumerate(res):
                for box in boxes:
                    box[2:4] = box[2:4] - box[:2]
                    p = {"image_id": i+1, "category_id": j+1, "bbox": box[:4].tolist(), "score": box[4]}
                    preds.append(p)
    if task == "V5M1280":
        with open(anno_path) as f:
            imgs = {
                os.path.basename(img_info["file_name"]): \
                {"id": img_info["id"], "w": img_info["width"], "h": img_info["height"]} \
                for img_info in json.load(f)["images"]
            }
        pred_dir = os.path.join(pred_path, "labels")
        for name, info in imgs.items():
            pred_path = os.path.join(pred_dir, name.replace(".jpg", ".txt"))
            if not os.path.exists(pred_path):
                continue
            with open(pred_path, 'r') as f:
                w, h, id = info["w"], info["h"], info["id"]
                for line in f.readlines():
                    instance = line.strip().split(" ")
                    b = list(map(float, instance[1:]))
                    box = [(b[0]-b[2]/2)*w, (b[1]-b[3]/2)*h, b[2]*w, b[3]*h]
                    p = {"image_id": id, "category_id": int(instance[0])+1, "bbox": box, "score": b[4]}
                    preds.append(p)
    return preds


def evaluate_by_coco(gt_path, preds):
    coco_gt = COCO(gt_path)
        
    coco_dt = coco_gt.loadRes(preds)
    # 初始化COCOeval对象
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.maxDets = [100, 300, 1000]
    # 运行评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 获取具体指标
    (mAP, AP50, mp), mr = coco_eval.stats[:3], coco_eval.stats[8]
    print(f'{mp:.3f}, {mr:9.3f}, {AP50:9.3f}, {mAP:9.3f}')



def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches)#.to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def evaluate_by_yolo(gt_path, preds):
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(preds)

    iouv = torch.linspace(0.5, 0.95, 10)
    stats, niou = [], iouv.numel()
    for image_id in coco_gt.getImgIds():#tqdm():
        ann_ids = coco_dt.getAnnIds(imgIds=image_id)
        pts = coco_dt.loadAnns(ann_ids)
        if len(pts) == 0:
            stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue
        boxes, pcls, pscores = zip(*([[anno['bbox'], anno['category_id'], anno['score']] for anno in pts]))
        pboxes = xywh2xyxy(np.array(boxes))
        pcls, pscores = np.array(pcls), np.array(pscores)
        predn = np.concatenate((pboxes, pscores.reshape([-1, 1]), pcls.reshape([-1, 1])), 1)

        ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        gts = coco_gt.loadAnns(ann_ids)
        if len(gts) == 0:
            correct = torch.zeros(predn.shape[0], niou, dtype=torch.bool)
            tcls = []
        else:
            boxes, tcls = zip(*([[anno['bbox'], anno['category_id']] for anno in gts]))
            tboxes = xywh2xyxy(np.array(boxes))
            tcls = np.array(tcls).reshape([-1, 1])
            labelsn = np.concatenate((tcls, tboxes), 1)
            correct = process_batch(torch.from_numpy(predn), torch.from_numpy(labelsn), iouv)
            tcls = tcls[:, 0].tolist()
        stats.append((correct, pscores, pcls, tcls))

    # {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=True, save_dir="runs/test", names={})
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    print(f'{mp:.3f}, {mr:9.3f}, {map50:9.3f}, {map:9.3f}')


if __name__=="__main__":
    # task = "V5M1280" # "JY-M" # "JY-M" #
    # pred_dir = "runs"
    # gt_file = "data/det-dataset/person/val.json"
    task = "CoDCN20973" #  
    pred_dir = "ckpts/private_handp/pretrained_by_coco"
    gt_file = "data/det-dataset/handp/test-20973.json"
    # gt_file = "data/det-dataset/handp/test-250320.json"
    preds = load_preds(pred_dir, gt_file, task)
    # evaluate_by_yolo(gt_file, preds)
    evaluate_by_coco(gt_file, preds)
