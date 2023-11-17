######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.16
# filenaem: vis_error.py
# function: visualize the error picture
######################################################
import argparse
import faiss
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.append('./')

from tools.post.feat_tools import load_data, choose_feats, create_index, save_keeps_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', "--gallerys", type=str)
    parser.add_argument('-q', "--querys", type=str)
    parser.add_argument('-l', "--label-file", type=str,
                        default='dataset/exp-data/zero_dataset/label_names.csv')
    parser.add_argument('-c', "--cats-file", type=str, 
                        default='dataset/exp-data/removeredundancy/4091_cats.txt')
    parser.add_argument('--pass-remove', action='store_true', default=False,
                        help="pass remove redundancy flag(True: pass, False: run remove)")
    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--save-root', type=str, default=False)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=9)
    return parser.parse_args()


def load_csv_file(label_file):
    product_id_map = {}
    with open(label_file) as f:
        for line in f:
            try:
                id_record = line.strip().replace('"', '').split(",")
                product_id_map[id_record[0]] = id_record[1]
            except:
                import pdb; pdb.set_trace()
    return product_id_map


def cv2AddChineseText(img, text, position, textColor, textSize):
    if (isinstance(img, np.ndarray)):  # OpenCV图片类型转为Image
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(
        "./dataset/exp-data/simsun.ttc", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw2big_pic(q_file, q_label, pred_files, pred_labels, label_map):
    sum_img = np.full((1282, 2408, 3), 255, dtype=np.uint8) #生成全白大图
    q_name = label_map[q_label]
    img = cv2.imread(q_file)
    color = (255, 0, 0)
    img = cv2AddChineseText(img, q_name, [2, 2], color, 16)
    H, W, _ = img.shape
    sum_img[0:H, 0:W, :] = img
    start_h, start_w = 0, 0
    for pred_file, pred_label in zip(pred_files, pred_labels):
        img = cv2.imread(pred_file)
        g_name = label_map[pred_label]
        color = (255, 0, 0) if q_label == pred_label else (0, 255, 0)
        img = cv2AddChineseText(img, g_name, [2, 2], color, 16)
        start_w += 482
        if start_w >= 2408:
            start_h += 642
            start_w = 0
        H, W, _ = img.shape
        new_height = int(H * (480 / W))  # 保持原来的长宽比  
        resized_img = cv2.resize(img, (480, new_height)) # 调整大小 
        # print('新的宽度和高度：', resized_img.shape[:2]) # 显示新的图像大小  

        sum_img[start_h:start_h+new_height, start_w:start_w+480, :] = resized_img
    return sum_img


def run_vis2bigimgs(initial_rank, g_labels, g_files, q_labels, q_files, label_map, save_root):
    for ind, (q_label, q_file) in enumerate(zip(q_labels, q_files)):
        q_name = label_map[q_label]
        obj_dir = os.path.join(save_root, q_name.split('/')[0])
        if not os.path.exists(obj_dir):
            os.mkdir(obj_dir)
        
        topks = initial_rank[ind]
        pred_files, pred_labels = g_files[topks], g_labels[topks]
        sum_img = draw2big_pic(q_file, q_label, pred_files, pred_labels, label_map)
        
        obj_path = os.path.join(obj_dir, q_file.split('/')[-1])
        cv2.imwrite(obj_path, sum_img)


if __name__ == '__main__':
    args = parse_args()
    param, measure = 'Flat', faiss.METRIC_INNER_PRODUCT
    # 加载npz文件
    gallery_feature, gallery_label, gallery_files = load_data(args.gallerys)
    query_feature, query_label, query_files = load_data(args.querys)
    with open(args.cats_file, 'r') as f: class_list = [line.strip('\n')[1:] for line in f.readlines()]
    faiss.normalize_L2(gallery_feature)

    if not args.pass_remove:
        keeps = choose_feats(gallery_feature, gallery_label, samilar_thresh=args.threshold, use_gpu=False)
        print(f"original data: {gallery_label.shape[0]}, after remove samilar data: {keeps.shape[0]}")
        new_g_feats = gallery_feature[keeps]
        new_g_label = gallery_label[keeps]
        new_g_files = gallery_files[keeps]
        np.savez(f'output/feats/regnety_040-train-{args.threshold}.npz', feats=new_g_feats, gts=new_g_label, fpaths=new_g_files)
    else:
        new_g_feats = gallery_feature
        new_g_label = gallery_label
        new_g_files = gallery_files
    
    # import pdb; pdb.set_trace()
    choose_cats = np.array([1210, 1447, 1521, 1991])
    cat_index = np.where(query_label[:, np.newaxis] == choose_cats)[0]
    query_feature = query_feature[cat_index]
    query_label = query_label[cat_index]
    query_files = query_files[cat_index]
    faiss.normalize_L2(query_feature)
    
    label_index = load_csv_file(args.label_file)
    cats = list(set(gallery_label))
    label_map = {i: label_index[cat] for i, cat in enumerate(class_list) if i in cats}
    index = create_index(new_g_feats, use_gpu=args.use_gpu)
    _, I = index.search(query_feature, args.topk)

    p_label = new_g_label[I]
    tp1_errors = np.where(p_label[:, 0] != query_label)[0]
    new_q_label, new_q_files = query_label[tp1_errors], query_files[tp1_errors]
    run_vis2bigimgs(I[tp1_errors], new_g_label, new_g_files, new_q_label, new_q_files, label_map, args.save_root)
