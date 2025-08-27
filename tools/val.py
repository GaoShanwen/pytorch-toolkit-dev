import argparse
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml) 

    model.info()

    model = YOLO('yolov8n.pt')

    model.val(
        data=opt.data,
        split='val',
        imgsz=opt.imgsz,
        batch=opt.batch,
        # channels=4,
        # use_simotm='RGBT',
        # rect=False,
        # save_json=True, # if you need to cal coco metrice
        # project='runs/val/LLVIP_r20',
        # name='LLVIP_r20-yolov8n-no_pretrained',
    )


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'yolov8n.pt', help='initial weights path')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)