######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.07.05
# filenaem: create_readable_dataset.py
# function: create readable dataset from raw dataset.
######################################################
import os
import argparse

from local_lib.data.readers.reader_image_in_txt import check_img

def parse_args():
    parser = argparse.ArgumentParser(description='Convert npz to bin')
    parser.add_argument('img_root', type=str, help='source file path')
    parser.add_argument('dst_file', type=str, help='destination root directory')
    return parser.parse_args()


def create_readable_dataset(img_root, dst_file):
    with open(dst_file, 'w') as f:
        for label in os.listdir(img_root):
            if not label.isdecimal():
                continue
            label_dir = os.path.join(img_root, label)
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if check_img(img_path):
                    f.write(f'{img_path},{label}\n')



if __name__ == '__main__':
    # set raw dataset path
    args = parse_args()
    create_readable_dataset(args.img_root, args.dst_file)