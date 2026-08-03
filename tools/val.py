######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.07.29
# filenaem: val.py
# function: validate dataset use yolo.
######################################################
from ultralytics import YOLO

import sys
sys.path.append('.')
from local_lib.utils import parse_args
from local_lib.models.val import CustomPoseValidator
from local_lib.models.symmetry_match.val import SymmetryMatchPoseValidator
from local_lib.models.mixed_data.val import MixedDataValidator
from local_lib.models.categorized_vis.val import CategorizedVisValidator



def get_validator(options):
    use_symmetry_match = options.pop("symmetry_match", False)
    use_mixed_data = options.pop("mixed_data", False)
    use_categorized_vis = options.pop("categorized_vis", False)

    if use_symmetry_match and use_mixed_data:
        return CustomPoseValidator
    if use_symmetry_match:
        return SymmetryMatchPoseValidator
    if use_mixed_data:
        return MixedDataValidator
    if use_categorized_vis:
        return CategorizedVisValidator
    return None


def validate(args):
    print(args)
    args.options = {} if args.options is None else args.options

    model = YOLO(model=args.model, task=args.task)

    model.val(
        validator=get_validator(args.options),
        data=args.data,
        split='val',
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        **args.options if args.options else {},
    )


if __name__ == "__main__":
    validate(parse_args())