import argparse
import os
import cv2
import time
import shutil 
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity
from multiprocessing import Pool, Manager, Lock, Process, cpu_count


def parse_args():
    parser = argparse.ArgumentParser(description='Download data from urls file')
    parser.add_argument('-i', '--img-root', type=str, default="data/det-dataset/allcar/250120", help='img root path')
    parser.add_argument('-d', '--obj-root', type=str, default="vis_imgs/similarities", help='obj root path or obj file')
    parser.add_argument('-s', '--size', type=int, default=(640,360), help="the size for compute similarty")
    parser.add_argument('-j', '--workers', type=int, default=48, help="the threshold for compute similarty")
    parser.add_argument('-th', '--threshold', type=float, default=0.9, help="the threshold for compute similarty")
    return parser.parse_args()


def compute_similarity(this_img, last_img):
    if last_img is None:
        return 0.
    return structural_similarity(this_img, last_img)


def compute_by_dir(src_paths, obj_dir, progress, saved, lock, _size=(640,360), th=0.9):
    last_img = last_path = None
    saved_num = 0
    saved_files = []
    try:
        for img_path in src_paths:
            current_img = cv2.imread(img_path, 0)
            resized_img = cv2.resize(current_img, _size)
            try:
                similarity = compute_similarity(resized_img, last_img)
            except ValueError as e:
                print(f"{e} when deal {last_path} with {img_path}!!!")
                continue
            if similarity > th:
                continue
            last_path = img_path
            last_img = resized_img
            saved_num += 1
            saved_files.append(img_path)
        if obj_dir.endswith(".txt"):
            with lock:
                with open(obj_dir, "a") as f:
                    for line in saved_files:
                        f.writelines(f"{line}\n")
        else:
            for img_path in saved_files:
                shutil.copy(img_path, obj_dir)

        saved.value += saved_num
        progress.value += len(src_paths)
    except Exception as e:
        raise Exception(f"{e} when deal {last_path} with {img_path}!!!")


def update_progress(progress, saved, total):
    last_value = 0
    prefix = "deal img... saved/finished/all="
    with tqdm(total=total, desc=f"{prefix}{saved.value}/{progress.value}/{total}", unit="element") as pbar:
        while progress.value < total:
            pbar.n = progress.value  # 设置当前进度
            pbar.last_print_n = progress.value  # 强制更新进度条
            pbar.set_description(f"{prefix}{saved.value}/{progress.value}/{total}")
            pbar.update(progress.value-last_value)
            last_value = progress.value
            time.sleep(0.1)
        pbar.set_description(f"{prefix}{saved.value}/{progress.value}/{total}")
        pbar.update(progress.value-last_value)


def do_remove(dst_dirs, obj_root, _size=(640, 360), workers=48, th=0.9):
    origin_paths = []
    for dst_dir in dst_dirs:
        paths = sorted([os.path.join(dst_dir, p) for p in os.listdir(dst_dir) if p.endswith(".jpg")])
        if len(paths):
            origin_paths.append(paths)
    origin_paths.sort(key=len, reverse=True)

    lock = Lock()
    with Manager() as manager:
        progress = manager.Value('i', 0)
        saved = manager.Value('i', 0)
        all_num = sum([len(ps) for ps in origin_paths])

        progress_thread = Process(target=update_progress, args=(progress, saved, all_num))
        progress_thread.start()

        processes = []
        for paths in origin_paths:
            if obj_root.endswith(".txt"):
                obj_dir = obj_root
            else: 
                obj_dir = os.path.join(obj_root, paths[0].split('/')[-2])
                if not os.path.exists(obj_dir):
                    os.mkdir(obj_dir)
            p = Process(target=compute_by_dir, args=(paths, obj_dir, progress, saved, lock, _size, th))
            p.start()
            processes.append(p)

            if len(processes) >= workers:
                processes[0].join()
                processes = processes[1:]
        
        for p in processes:
            p.join()
        progress_thread.join()
        print(f"save/all:{saved.value}/{all_num}={saved.value/max(1, all_num)*100:.02f}%")


if __name__=="__main__":
    args = parse_args()
    img_root = args.img_root
    dst_dirs = [root for root, _, _ in os.walk(img_root)]

    if not args.obj_root.endswith(".txt") and not os.path.exists(args.obj_root):
        os.makedirs(args.obj_root)
    do_remove(dst_dirs, args.obj_root, args.size, args.workers, args.threshold)
    # with open(args.obj_root, "r") as f:
    #     choices = np.array([i.strip() for i in f.readlines()])
    # with open("data/det-dataset/hld/train250116.txt", "r") as f:
    #     trainset = np.array([i.strip() for i in f.readlines()])
    # keeps = np.isin(choices, trainset)
    # with open("data/det-dataset/hld/train.txt", "w") as f:
    #     for i in choices[keeps]:
    #         f.writelines(f"{i}\n")
