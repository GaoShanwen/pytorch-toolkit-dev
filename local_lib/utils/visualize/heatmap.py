######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: vis_heatmap.py
# function: visual the heatmap by model.
######################################################
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import AblationCAM, EigenCAM, FullGrad, GradCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM
from pytorch_grad_cam.utils.image import deprocess_image, preprocess_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from timm.models import load_checkpoint
from timm.utils import ParseKwargs

from local_lib.models import create_custom_model

parser = argparse.ArgumentParser(description="PyTorch ImageNet Validation")
parser.add_argument("output", metavar="save_root", help="output picture root")
parser.add_argument("--num-classes", type=int, default=None, help="Number classes in dataset")
parser.add_argument(
    "--model",
    metavar="NAME",
    default="dpn92",
    help="model architecture (default: dpn92)",
)
parser.add_argument(
    "--checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--use-ema",
    dest="use_ema",
    action="store_true",
    help="use ema version of weights if present",
)
parser.add_argument("--model-kwargs", nargs="*", default={}, action=ParseKwargs)


# 定义CAM可视化函数
def visualize_CAM(model, model_name, img_path, save_root):
    image = cv2.imread(img_path, 1)  # imread()读取的是BGR格式
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    rgb_img = np.float32(image) / 255.0
    # preprocess_image作用：归一化图像，并转成tensor
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 3)初始化CAM对象，包括模型，目标层以及是否使用cuda等
    # Construct the CAM object once, and then re-use it on many images:
    if "regnety" in model_name:
        target_layer = [model.s4]
    elif "mobilenetv3" in model_name:
        target_layer = [model.blocks[-1]]
    else:
        raise f"{model_name} not be support!"
    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)
    # 4)选定目标类别，如果不设置，则默认为分数最高的那一类
    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    target_category = None
    # 指定类：target_category = 281

    # 5)计算cam
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)  # [batch, 224,224]

    # 6)展示热力图并保存
    # In this example grayscale_cam has only one image in the batch:
    # 7.展示热力图并保存, grayscale_cam是一个batch的结果，只能选择一张进行展示
    visualization = show_cam_on_image(rgb_img, grayscale_cam[0])  # (224, 224, 3)
    cv2.imwrite(os.path.join(save_root, "first_try.jpg"), visualization)


if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.model
    model = create_custom_model(
        model_name, num_classes=args.num_classes, in_chans=3, global_pool=None, **args.model_kwargs
    )
    load_checkpoint(model, args.checkpoint, args.use_ema)

    # 加载一张测试图像
    img_path = "dataset/test_imgs/1.jpg"
    visualize_CAM(model, model_name, img_path, args.output)
