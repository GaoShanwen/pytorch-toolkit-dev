import argparse
import sys
import cv2
import os
import shutil
from ultralytics import YOLO
from ultralytics.models.yolo.pose import PosePredictor

sys.path.append('.')
from local_lib.models import YOLOPro

def parse_args():
    parser = argparse.ArgumentParser(description='YolLOv8 Pose Inference')
    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights file')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image or video file')
    parser.add_argument('--symmetry-match', action='store_true', default=False, help='Whether to use symmetry match')
    parser.add_argument('--mix-data', action='store_true', default=False, help='Whether to use mixed data')
    parser.add_argument('--quad-eiou', action='store_true', default=False, help='Whether to use quad EIoU')
    parser.add_argument('--save', action='store_true', default=True, help='Save results')
    parser.add_argument('--flip', action='store_true', default=False, help='Enable horizontal flip inference')
    return parser.parse_args()


def predict(args):
    args = parse_args()
    print(args)
    # model_name = YOLOPro if args.symmetry_match or args.mix_data or args.quad_eiou else YOLO
    Predictor = PosePredictor if args.symmetry_match or args.mix_data or args.quad_eiou else None
    model = YOLO(args.weights)
    
    print("=== 原始图像推理 ===")
    model.predict(args.img_path, save=args.save, predictor=Predictor)
    
    if args.flip:
        task = args.weights.split("/")[1]
        os.makedirs(f"runs/{task}/predict2", exist_ok=True)

        for img_file in os.listdir(f"runs/{task}/predict2"):
            os.remove(os.path.join(f"runs/{task}/predict2", img_file))

        print("\n=== 水平翻转图像推理 ===")
        for img_name in os.listdir(args.img_path):
            img = cv2.imread(os.path.join(args.img_path, img_name))
            flipped_img = cv2.flip(img, 1)
            
            model.predict(flipped_img, save=args.save, predictor=Predictor)
            shutil.move(os.path.join(f"runs/{task}/predict", "image0.jpg"), os.path.join(f"runs/{task}/predict2", img_name))


if __name__ == '__main__':
    predict(parse_args())