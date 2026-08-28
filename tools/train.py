#!/usr/bin/env python
"""
RF-DETR training entry point.

Usage:
    python tools/train.py --data <dataset_dir> --epochs 100 --batch 4 \\
        [--class_mapping 5:0,6:1,7:2] [--mixed_alpha 0.06] \\
        [--background_data data/background.txt]
"""
import sys
sys.path.insert(0, ".")

from local_lib.utils import parse_args
from local_lib.models.train import train


if __name__ == "__main__":
    train(parse_args())
