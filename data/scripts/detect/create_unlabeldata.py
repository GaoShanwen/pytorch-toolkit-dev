import os
import shutil
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("visualize boxes parameters")
    parser.add_argument("-i", "--img-root", type=str, required=True, default=None)
    parser.add_argument("-d", "--dst-root", type=str, required=True, default=None)
    parser.add_argument("-g", "--gap", type=int, default=3)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    src_dirs = [root for root, _, _ in os.walk(args.img_root)]
    choose = num = 0
    for src_dir in src_dirs:
        paths = sorted([os.path.join(src_dir, p) for p in os.listdir(src_dir) if p.endswith(".jpg")])
        dst_dir = src_dir.replace(args.img_root, args.dst_root)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        for image_path in paths:
            choose += 1
            if choose % args.gap:
                continue
            shutil.copy(image_path, dst_dir)
            label_path = image_path[:-4] + ".txt"

            if not os.path.exists(label_path): # not os.path.exists(image_path) or 
                # print(f"Error: Label file not found for {label_path}")
                continue
            shutil.copy(label_path, dst_dir)
        num += len(paths)
    print(f"choose/all num: {choose//3}/{num}")
        