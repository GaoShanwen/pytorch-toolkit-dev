from ultralytics import YOLO
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train/Validate a model")
    parser.add_argument("-t", "--task", type=str, required=True, help="train task(such as mandp/candp)")
    # parser.add_argument("--num-classes", type=int, default=3, help="train task")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    src_root = f"ckpts/{args.task}"
    model_dirs = os.listdir(src_root)
    assert len(model_dirs), "Couldn't find model directory in %s" % src_root
    model_version = sorted(model_dirs)[-1]
    src_dir = os.path.join(src_root, model_version)
    model = YOLO(os.path.join(src_dir, 'weights/last.pt'), task="detect")

    obj_path = f"ckpts/{args.task}_{model_version}.onnx"
    print(model_dirs, f"saved in {obj_path}")
    path = model.export(format="onnx", simplify=True, device=0, opset=12, dynamic=False, imgsz=640)
    os.path.rename(path, obj_path)
