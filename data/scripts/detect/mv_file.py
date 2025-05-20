import os
import random
import shutil
import cv2
from tqdm import tqdm


def check_img(filename):
    if not os.path.exists(filename):
        return False
    try:
        with open(filename, "rb") as f:
            f.seek(-2, 2)
            if not f.read() == b"\xff\xd9":
                return False
    except IOError:
        return False
    return True


if __name__ == '__main__':
    img_files = "data/det-dataset/CoalCutter/total_train.txt"
    with open(img_files, "r") as f:
        file_paths = [line.strip() for line in f.readlines()]
    for file_path in file_paths:#tqdm():
        if not file_path.endswith(".jpeg"):
            continue
        img = cv2.imread(file_path)
        if not check_img(file_path):
            print(file_path)
        # cv2.imwrite(file_path.replace(".jpeg", ".jpg"), img)