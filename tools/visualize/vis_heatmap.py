######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: vis_heatmap.py
# function: visual the heatmap by model.
######################################################
import argparse
import torch
import cv2
import torch.nn as nn
import torchvision.transforms as transforms
from timm.models import create_model, load_checkpoint

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    deprocess_image,
    preprocess_image,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from timm.utils import ParseKwargs

import sys

sys.path.append("./")

import local_lib.models

parser = argparse.ArgumentParser(description="PyTorch ImageNet Validation")
parser.add_argument("output", metavar="save_root", help="output picture root")
parser.add_argument("--num-classes", type=int, default=None, help="Number classes in dataset")
parser.add_argument(
    "--checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--model",
    "-m",
    metavar="NAME",
    default="dpn92",
    help="model architecture (default: dpn92)",
)
parser.add_argument(
    "--use-ema",
    dest="use_ema",
    action="store_true",
    help="use ema version of weights if present",
)
parser.add_argument("--model-kwargs", nargs="*", default={}, action=ParseKwargs)


# 定义CAM可视化函数
def visualize_CAM(image, model):
    rgb_img = np.float32(image) / 255.0
    # preprocess_image作用：归一化图像，并转成tensor
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ----------------------------------------
    """
    3)初始化CAM对象，包括模型，目标层以及是否使用cuda等
    """
    # Construct the CAM object once, and then re-use it on many images:
    target_layer = [model.blocks[-1]]
    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)
    """
    4)选定目标类别，如果不设置，则默认为分数最高的那一类
    """
    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    target_category = None
    # 指定类：target_category = 281

    """
    5)计算cam
    """
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)  # [batch, 224,224]

    """
    6)展示热力图并保存
    """
    # In this example grayscale_cam has only one image in the batch:
    # 7.展示热力图并保存, grayscale_cam是一个batch的结果，只能选择一张进行展示
    visualization = show_cam_on_image(rgb_img, grayscale_cam[0])  # (224, 224, 3)
    cv2.imwrite(f"first_try.jpg", visualization)


if __name__ == "__main__":
    args = parser.parse_args()
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=None,
        scriptable=False,
        **args.model_kwargs,
    )
    load_checkpoint(model, args.checkpoint, args.use_ema)
    # resize操作是为了和传入神经网络训练图片大小一致
    # img = Image.open(img_path)#.resize((224,224))
    # 需要将原始图片转为np.float32格式并且在0-1之间
    # rgb_img = np.float32(img)/255
    # means=[0.4850, 0.4560, 0.4060]
    # std=[0.2290, 0.2240, 0.2250]
    # img = img/255.
    # img = (img-means)/std
    # img = img.transpose((2,0,1))[np.newaxis, ...]
    # plt.imshow(img)

    # 加载一张测试图像
    img_path = "dataset/exp-data/minidata/quantizate/100_NZ53MZV0KS_1680344371005_1680344371719.jpg"
    # image = plt.imread(img_path)
    rgb_img = cv2.imread(img_path, 1)  # imread()读取的是BGR格式
    # 可视化CAM
    visualize_CAM(rgb_img, model)
    # import pdb; pdb.set_trace()
    # target_layers = [model.blocks[-1]] #[model.features[-1]]
    # # # 选取合适的类激活图，但是ScoreCAM和AblationCAM需要batch_size
    # cam = GradCAM(model=model, target_layers=target_layers)
    # # targets = [ClassifierOutputTarget(preds)]
    # # 上方preds需要设定，比如ImageNet有1000类，这里可以设为200
    # grayscale_cam = cam(input_tensor=torch.from_array(img), targets=targets)
    # grayscale_cam = grayscale_cam[0, :]
    # cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # print(type(cam_img))
    # Image.fromarray(cam_img)
