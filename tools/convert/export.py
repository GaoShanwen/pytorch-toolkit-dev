import argparse
import torch

from ultralytics import YOLO

import sys
sys.path.append('.')
from local_lib.models import YOLOPro

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-path', type=str, required=True, help='weight path')
    parser.add_argument('--image-path', type=str, default="data/pose-dataset/Person/demo.jpeg", help='weight path')
    parser.add_argument('--symmetry-match', action='store_true', default=False, help='Whether to use symmetry match')

    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    # Load a pretrained YOLO26n model 
    model_name = YOLOPro if args.symmetry_match else YOLO
    model = model_name(args.weight_path, task="pose")   

    # # Evaluate the model's performance on the validation set
    # metrics = model.val()

    # # Perform object detection on an image
    # results = model(args.image_path)  # Predict on an image
    # results[0].show()  # Display results

    # Export the model to ONNX format for deployment
    model.eval()
    with torch.no_grad(): # 必须关闭梯度
        path = model.export(format="onnx", imgsz=[384, 640])  # Returns the path to the exported model
