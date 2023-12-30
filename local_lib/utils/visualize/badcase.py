######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.16
# filenaem: badcase.py
# function: visualize the error picture
######################################################
import os
import shutil

import cv2
import numpy as np
import tqdm
from PIL import Image, ImageDraw, ImageFont

from local_lib.utils.feat_tools import create_index


def vis_text(img, text, position, text_color, text_size):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("dataset/simsun.ttc", text_size, encoding="utf-8")
    draw.text(position, text, text_color, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw2big_pic(q_file, q_label, pred_files, pred_labels, label_map, text_size=24, scores=None):
    sum_img = np.full((1282, 2408, 3), 255, dtype=np.uint8)  # 生成全白大图
    q_name = label_map[q_label] if label_map is not None else str(q_label)
    if not os.path.exists(q_file):
        return None
    img = cv2.imread(q_file)
    color = (255, 0, 0)
    H, W, _ = img.shape
    new_height = min(640, int(H * (480 / W)))  # 保持原来的长宽比
    img = cv2.resize(img, (480, new_height))  # 调整大小
    img = vis_text(img, q_name, [2, 2], color, text_size)
    sum_img[0:new_height, 0:480, :] = img
    start_h, start_w = 0, 0
    for i, (pred_file, pred_label) in enumerate(zip(pred_files, pred_labels)):
        img = cv2.imread(pred_file) if os.path.exists(pred_file) else np.ones((640, 480), dtype=np.uint8) * 255
        new_height = min(640, int(H * (480 / W)))  # 保持原来的长宽比
        img = cv2.resize(img, (480, new_height))  # 调整大小
        g_name = label_map[pred_label] if label_map is not None else str(pred_label)
        color = (255, 0, 0) if q_label == pred_label else (0, 255, 0)
        img = vis_text(img, g_name, [2, 2], color, text_size)
        H, W, _ = img.shape
        if scores is not None:
            img = vis_text(img, f"{scores[i]:.5f}", [2, H - text_size - 2], color, text_size)
        start_w += 482
        if start_w >= 2408:
            start_h += 642
            start_w = 0
        sum_img[start_h : start_h + new_height, start_w : start_w + 480, :] = img
    return sum_img


def run_vis2bigimgs(
    initial_rank, g_labels, g_files, q_labels, q_files, label_map, save_root, text_size=24, scores=None
):
    pbar = tqdm.tqdm(total=q_labels.shape[0])
    for ind, (q_label, q_file) in enumerate(zip(q_labels, q_files)):
        pbar.update(1)
        q_name = label_map[q_label] if label_map is not None else str(q_label)
        obj_dir = os.path.join(save_root, q_name.split("/")[0])
        if not os.path.exists(obj_dir):
            os.mkdir(obj_dir)

        topks = initial_rank[ind]
        pred_files, pred_labels = g_files[topks], g_labels[topks]
        index_scores = scores[ind] if scores is not None else None
        sum_img = draw2big_pic(q_file, q_label, pred_files, pred_labels, label_map, text_size, scores=index_scores)
        if sum_img is None:
            continue
        obj_path = os.path.join(obj_dir, q_file.split("/")[-1])
        cv2.imwrite(obj_path, sum_img)
    pbar.close()


def save_imgs(files, labels, class_list, save_root):
    with open("/".join(save_root.split("/")[:-2] + ["choose_noise.txt"]), "w") as f:
        for i, (current_file, current_label) in enumerate(zip(files, labels)):
            label_name = class_list[current_label] if class_list is not None else str(current_label)
            obj_dir = os.path.join(save_root, label_name)
            if not os.path.exists(obj_dir):
                os.mkdir(obj_dir)
            obj_path = os.path.join(obj_dir, f"{i:08d}.jpg")
            shutil.copy(current_file, obj_path)
            f.write(f"{current_file},{obj_path}\n")


def search_and_vis(g_feats, g_label, g_files, q_feats, q_label, q_files, args, label_map=None):
    index = create_index(g_feats, use_gpu=args.use_gpu, param=args.param, measure=args.measure)
    S, I = index.search(q_feats, args.topk)

    p_label = g_label[I]
    tp1_errors = np.where(p_label[:, 0] != q_label)[0]  # np.array(np.arange(q_label.shape[0]))#
    print(f"choose {tp1_errors.shape[0]} infer errors from {q_label.shape[0]} imgs")
    q_label, q_files = q_label[tp1_errors], q_files[tp1_errors]
    new_idx, new_scores = I[tp1_errors], S[tp1_errors]
    save_root, text_size = args.save_root, args.text_size
    run_vis2bigimgs(new_idx, g_label, g_files, q_label, q_files, label_map, save_root, text_size, new_scores)
