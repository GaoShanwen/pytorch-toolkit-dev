from ultralytics import YOLO
import sys
sys.path.append('.')
from local_lib.models import YOLOPro
import os
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train/Validate a model")
    parser.add_argument("-t", "--task", type=str, required=True, help="train task(such as mandp/candp)")
    parser.add_argument("-f", "--format", type=str, required=True, help="export format(such as onnx/engine)")
    # parser.add_argument("--num-classes", type=int, default=3, help="train task")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    src_root = f"ckpts/{args.task}"
    model_dirs = [name for name in os.listdir(src_root) if not name.startswith("V5")]
    assert len(model_dirs), "Couldn't find model directory in %s" % src_root
    model_version = sorted(model_dirs)[-1]
    src_dir = os.path.join(src_root, model_version)
    print(f"load model from {src_dir}")
    model_name = YOLOPro if args.task == "vehicle" else YOLO
    model = model_name(os.path.join(src_dir, 'weights/best.pt'), task="detect")

    obj_path = f"tools/convert/model/{args.task}_{model_version}.{args.format}"
    path = model.export(format=args.format, simplify=True, device=0, opset=12, dynamic=False, imgsz=640)
    # print(path, obj_path)
    shutil.move(path, obj_path)
    print(model_dirs, f"saved in {obj_path}")
