######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.16
# filenaem: vis_error.py
# function: visualize the error picture
######################################################
import os

import cv2
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_RED = (255, 0, 0)
_GREEN = (0, 255, 0)

def vis_text(img, text, position, text_color, text_size):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("dataset/simsun.ttc", text_size, encoding="utf-8")
    draw.text(position, text, text_color, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def save_imgs(choose_files, choices_type, save_root):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for file_path, cat in zip(choose_files, choices_type):
        save_dir = os.path.join(save_root, f"{cat}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        shutil.copy(file_path, save_dir)


class VisualizeResults:
    def __init__(
        self, save_root, task="classify", text_size=24, class_map=None, settings=[], img_size=(640, 480)
    ) -> None:
        assert task in ["classify", "search"], f"{task} is not support, only support classify or search, yet!"
        assert all(isinstance(v, str) for v in class_map.values()), "the values of class_map must be str!"
        self.vis_function = eval(f"self.vis_one_{task}")
        self.only_error = "only_error" in settings
        self.show_gt = "show_gt" in settings
        self.save_root = save_root
        self.img_size = img_size
        self.text_size = text_size
        self.class_map = class_map

        rows, columns, edge = 2, 5, 2
        set_height, set_width = img_size
        self.strip_h, self.strip_w = set_height + edge, set_width + edge
        self.output_height = (set_height + edge) * rows - edge
        self.output_width = (set_width + edge) * columns - edge

    def vis_one_search(self, q_name, q_file, pred_names, pred_files, scores=None):
        sum_img = np.full((self.output_height, self.output_width, 3), 255, dtype=np.uint8)  # 生成全白大图
        set_height, set_width = self.img_size

        img = cv2.imread(q_file)
        H, W, _ = img.shape
        new_height = min(set_height, int(H * (set_width / W)))  # 保持原来的长宽比
        img = vis_text(img, q_name, [2, 2], _RED, self.text_size)
        sum_img[0:new_height, 0:set_width, :] = img
        start_h, start_w, default_sw = 0, 0, 0
        if self.show_gt:
            if q_name in pred_names:
                gt_idx = pred_names.index(q_name)
                gt_files = pred_files[gt_idx]
                img = cv2.imread(gt_files) if os.path.exists(gt_files) else np.full(self.img_size, 255, dtype=np.uint8)
                H, W, _ = img.shape
                new_height = min(set_height, int(H * (set_width / W)))  # 保持原来的长宽比
                text = f"第{len(set(pred_names[:gt_idx]))+1}类; 第{gt_idx+1}张"
                img = vis_text(img, text, [2, 2], _RED, self.text_size)
                text = f"{scores[gt_idx]:.5f}" if scores is not None else ""
                img = vis_text(img, text, [2, new_height - self.text_size - 2], _RED, self.text_size)
                sum_img[self.strip_h : self.strip_h + new_height, start_w : start_w + set_width, :] = img
            default_sw = self.strip_w
        show_num = 8 if self.show_gt else 9
        for i, (pred_file, p_name) in enumerate(zip(pred_files[:show_num], pred_names[:show_num])):
            img = cv2.imread(pred_file) if os.path.exists(pred_file) else np.full(self.img_size, 255, dtype=np.uint8)
            H, W, _ = img.shape
            new_height = min(set_height, int(H * (set_width / W)))  # 保持原来的长宽比
            img = cv2.resize(img, (set_width, new_height))  # 调整大小
            color = _RED if q_name == p_name else _GREEN
            img = vis_text(img, p_name, [2, 2], color, self.text_size)
            if scores is not None:
                img = vis_text(img, f"{scores[i]:.5f}", [2, new_height - self.text_size - 2], color, self.text_size)
            start_w += self.strip_w
            if start_w >= self.output_width:
                start_h += self.strip_h
                start_w = default_sw
            sum_img[start_h : start_h + new_height, start_w : start_w + set_width, :] = img
        return sum_img

    def vis_one_classify(self, q_name, q_file, pred_names, pred_files, scores=None):
        set_height, set_width = self.img_size
        img = cv2.imread(q_file)
        H, W, _ = img.shape
        new_height = min(set_height, int(H * (set_width / W)))  # 保持原来的长宽比
        img = cv2.resize(img, (set_width, new_height))  # 调整大小
        assert len(pred_names) == len(scores), f"pred name number is {len(pred_names)}, scores is {len(scores)}"
        for i, (p_name, score) in enumerate(zip(pred_names, scores)):
            img = vis_text(img, p_name, [2, i * self.text_size + 2], (255, 0, 0), self.text_size)
            if score is not None:
                start_h = H // 2 + i * self.text_size + 2
                img = vis_text(img, f"{score*100:.2f}", [2, start_h], (255, 0, 0), self.text_size)
        return img

    def get_save_dir(self, obj_file, class_name):
        file_name = os.path.basename(obj_file)
        obj_dir = os.path.join(self.save_root, class_name)
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)
        return os.path.join(obj_dir, file_name)

    def do_visualize(self, gt_labels, gt_files, p_labels, p_files=None, scores=None) -> np.ndarray:
        if p_files is None:
            p_files = np.full(gt_labels.shape[0], np.nan)
        if scores is None:
            scores = np.full(gt_labels.shape[0], np.nan)
        for gt_label, gt_file, p_label, p_file, score in zip(gt_labels, gt_files, p_labels, p_files, scores):
            if self.only_error and gt_label in p_label[:5]:#p_label[:5]:
                continue
            if not os.path.exists(gt_file):
                print(f"{gt_file} is error!")
                continue
            if self.class_map is not None:
                gt_label = self.class_map[gt_label]
                p_label = [self.class_map[p] for p in p_label if p in self.class_map]
                score = score[:len(p_label)]
            img = self.vis_function(gt_label, gt_file, p_label, p_file, score)
            obj_path = self.get_save_dir(gt_file, gt_label)
            cv2.imwrite(obj_path, img)
