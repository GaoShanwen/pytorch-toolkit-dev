######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2025.08.27
# filenaem: val.py
# function: validate dataset use yolo.
######################################################
from ultralytics import YOLO
import sys
sys.path.append('.')
from local_lib.utils.set_parse import parse_args
from local_lib.models.det_ignore import WithIgnoreValidator

def validate(args):
    print(args)
    model = YOLO(model=args.model, task=args.task)
    if args.options.pop("with_ignore", False):
        args.options.update({"validator": WithIgnoreValidator})
    model.val(
        data=args.data,
        split='val',
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        **args.options if args.options else {},
    )


if __name__ == "__main__":
    validate(parse_args())