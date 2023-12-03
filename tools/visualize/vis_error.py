######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.16
# filenaem: vis_error.py
# function: visualize the error picture
######################################################
import argparse
import shutil
import faiss
import os
import cv2
import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.append('./')

from tools.post.feat_tools import load_data, create_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', "--gallerys", type=str)
    parser.add_argument('-q', "--querys", type=str)
    parser.add_argument('-l', "--label-file", type=str,
                        default='dataset/exp-data/zero_dataset/label_names.csv')
    parser.add_argument('-c', "--cats-file", type=str, 
                        default='dataset/exp-data/removeredundancy/629_cats.txt')
    parser.add_argument('--pass-remove', action='store_true', default=False,
                        help="pass remove redundancy flag(True: pass, False: run remove)")
    parser.add_argument("--pass-cats", type=str, 
                        default='dataset/exp-data/removeredundancy/pass_cats.txt')
    parser.add_argument("--mask-path", type=str, default='blacklist-val.npy')
    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--save-root', type=str, default='output/vis/errors')
    parser.add_argument("--num-classes", type=int, default=629)
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


def cv2AddChineseText(img, text, position, text_color, text_size):
    if (isinstance(img, np.ndarray)):  # OpenCV图片类型转为Image
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(
        "./dataset/exp-data/simsun.ttc", text_size, encoding="utf-8")
    draw.text(position, text, text_color, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw2big_pic(q_file, q_label, pred_files, pred_labels, label_map, text_size=24, scores=None):
    sum_img = np.full((1282, 2408, 3), 255, dtype=np.uint8) #生成全白大图
    q_name = label_map[q_label] if label_map is not None else str(q_label)
    img = cv2.imread(q_file)
    color = (255, 0, 0)
    img = cv2AddChineseText(img, q_name, [2, 2], color, text_size)
    H, W, _ = img.shape
    sum_img[0:H, 0:W, :] = img
    start_h, start_w = 0, 0
    for i, (pred_file, pred_label) in enumerate(zip(pred_files, pred_labels)):
        img = cv2.imread(pred_file)
        g_name = label_map[pred_label] if label_map is not None else str(pred_label)
        color = (255, 0, 0) if q_label == pred_label else (0, 255, 0)
        img = cv2AddChineseText(img, g_name, [2, 2], color, text_size)
        H, W, _ = img.shape
        if scores is not None:
            img = cv2AddChineseText(img, f"{scores[i]}", [2, H-text_size-2], color, text_size)
        start_w += 482
        if start_w >= 2408:
            start_h += 642
            start_w = 0
        new_height = min(640, int(H * (480 / W)))  # 保持原来的长宽比  
        resized_img = cv2.resize(img, (480, new_height)) # 调整大小 
        # print('新的宽度和高度：', resized_img.shape[:2]) # 显示新的图像大小  

        sum_img[start_h:start_h+new_height, start_w:start_w+480, :] = resized_img
    return sum_img


def run_vis2bigimgs(initial_rank, g_labels, g_files, q_labels, q_files, label_map, save_root, text_size=24, scores=None):
    pbar = tqdm.tqdm(total=q_labels.shape[0])
    for ind, (q_label, q_file) in tqdm.tqdm(enumerate(zip(q_labels, q_files))):
        q_name = label_map[q_label] if label_map is not None else str(q_label)
        obj_dir = os.path.join(save_root, q_name.split('/')[0])
        if not os.path.exists(obj_dir):
            os.mkdir(obj_dir)
        
        topks = initial_rank[ind]
        pred_files, pred_labels = g_files[topks], g_labels[topks]
        index_scores = scores[ind] if scores is not None else None
        sum_img = draw2big_pic(q_file, q_label, pred_files, pred_labels, label_map, text_size=text_size, scores=index_scores)
        
        obj_path = os.path.join(obj_dir, q_file.split('/')[-1])
        cv2.imwrite(obj_path, sum_img)
        pbar.update(1)
    pbar.close()


def save_imgs(labels, files, class_list, save_root):
    with open('/'.join(save_root.split('/')[:-2] + ['choose_noise.txt']), 'w') as f:
        for i, (current_file, current_label) in enumerate(zip(files, labels)):
            label_name = class_list[current_label]
            obj_dir = os.path.join(save_root, label_name)
            if not os.path.exists(obj_dir):
                os.mkdir(obj_dir)
            obj_path = os.path.join(obj_dir, f"{i:08d}.jpg")
            shutil.copy(current_file, obj_path)
            f.write(f"{current_file},{obj_path}\n")


if __name__ == '__main__':
    args = parse_args()
    # param, measure = 'Flat', faiss.METRIC_INNER_PRODUCT
    args.param = f'IVF{args.num_classes},Flat'
    args.measure = faiss.METRIC_INNER_PRODUCT
    # 加载npz文件
    gallery_feature, gallery_label, gallery_files = load_data(args.gallerys)
    query_feature, query_label, query_files = load_data(args.querys)
    with open(args.cats_file, 'r') as f: class_list = [line.strip('\n') for line in f.readlines()]
    faiss.normalize_L2(gallery_feature)
    faiss.normalize_L2(query_feature)

    if args.debug:
        # with open(args.pass_cats, 'r') as f: mask_cats = [line.strip('\n') for line in f.readlines()]
        # choose_cats = list(set(gallery_label))
        # # print(class_list, len(choose_cats))
        # # choose_cats = np.array([class_list.index(cat) for cat in ["999920265", "10000000105", "999920247", "9999150390"]])
        # # # choose_cats = np.array([1210, 1447, 1521, 1991])
        # for cat in mask_cats:
        #     choose_cats.remove(class_list.index(cat))
        # # choose_cats = list(set(g_label))[:15]
        # cat_index = np.where(gallery_label[:, np.newaxis] == choose_cats)[0]
        # gallery_feature, gallery_label, gallery_files = gallery_feature[cat_index], gallery_label[cat_index], gallery_files[cat_index]

        # cat_index = np.where(query_label[:, np.newaxis] == choose_cats)[0]
        # query_feature, query_label, query_files = query_feature[cat_index], query_label[cat_index], query_files[cat_index]
    
        mask_files = np.load(args.mask_path)
        masks = np.isin(query_files, mask_files)
        keeps = np.array(np.arange(query_files.shape[0]))[~masks]
        query_feature, query_label, query_files =  query_feature[keeps], query_label[keeps], query_files[keeps]
    label_index = load_csv_file(args.label_file)
    cats = list(set(gallery_label))
    label_map = {i: label_index[cat] for i, cat in enumerate(class_list) if i in cats}
    index = create_index(gallery_feature, use_gpu=args.use_gpu)
    S, I = index.search(query_feature, args.topk)

    p_label = gallery_label[I]
    tp1_errors = np.where(p_label[:, 0] != query_label)[0]
    # tp1_errors = []
    # for i, p in enumerate(p_label):
    #     black_num = np.where(p==9)[0].shape[0]
    #     save_cats = len(set(p))
    #     if not black_num and save_cats==1:
    #         tp1_errors.append(i)
    # tp1_errors = np.where(query_label!=9)[0] #np.array(np.arang(query_label.shape[0]))
    # label_map = None
    # import pdb; pdb.set_trace()
    new_q_label, new_q_files = query_label[tp1_errors], query_files[tp1_errors]
    run_vis2bigimgs(I[tp1_errors], gallery_label, gallery_files, new_q_label, new_q_files, label_map, args.save_root, text_size=32, scores=S)
