#!/usr/bin/env python3
######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.09
# filenaem: create_lists.py
# function: build the yolo-style train / val / test txt lists of a seg dataset.
######################################################
import os
import sys
import argparse
import logging

# make the repo root importable when running as `python tools/create_lists.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_lib.data import create_dataset_lists

_logger = logging.getLogger("create_lists")


def parse_args():
    parser = argparse.ArgumentParser(
        "Build the yolo-style train/val/test txt lists of a segmentation dataset."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="dataset root holding the 'images' and 'labels' folders",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="where the txt lists are written (defaults to the dataset root)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="share used for training")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="share used for validation")
    parser.add_argument("--seed", type=int, default=42, help="seed of the shuffle")
    parser.add_argument(
        "--images-dir-name",
        type=str,
        default="images",
        help="folder name of the images, replaced by the labels folder name",
    )
    parser.add_argument(
        "--labels-dir-name",
        type=str,
        default="labels",
        help="folder name of the masks",
    )
    parser.add_argument("--mask-ext", type=str, default=".png", help="extension of the masks")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    result = create_dataset_lists(
        args.root,
        out_dir=args.out_dir or None,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        images_dir_name=args.images_dir_name,
        labels_dir_name=args.labels_dir_name,
        mask_ext=args.mask_ext,
    )
    for name, list_file in result["lists"].items():
        _logger.info(f"{name}: {list_file} ({result['counts'][name]} images)")


if __name__ == "__main__":
    main()
