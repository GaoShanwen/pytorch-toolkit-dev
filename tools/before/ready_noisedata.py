import os
import shutil
import glob
import tqdm
import numpy as np


def do_write(src_file, add_file, pass_file, obj_file):
    with open(pass_file, "r") as f:
        pass_imgs = [line.strip('\n').split(",")[0] for line in f.readlines()]
    with open(src_file, "r") as f:
        src_connets = [line.strip('\n').split(", ") for line in f.readlines()]
    with open(add_file, "r") as f:
        add_connets = [line.strip('\n').split(", ") for line in f.readlines()]
    src_imgs = np.array([img for img, _ in src_connets])
    masks = np.isin(src_imgs, pass_imgs)
    src_connets = np.array(src_connets)[~masks]
    with open(obj_file, "w") as f:
        pbar = tqdm.tqdm(total=len(add_connets)+src_connets.shape[0])#len(add_connets)+len(src_connets))
        for (src_img, label) in add_connets:
            pbar.update(1)
            if label == "9":
                continue
            f.write(f"{src_img}, {int(label):08d}\n")
        # strat_p = 0
        for (src_img, label) in src_connets:
            pbar.update(1)
            # if src_img in pass_imgs[strat_p:]:
            #     strat_p += pass_imgs[strat_p:].index(src_img)
            #     continue
            f.write(f"{src_img}, {label}\n")


if __name__ == "__main__":
    # data_root = "output/vis/noises"
    # obj_root = "output/vis/need_label"

    # jpg_files = glob.glob(os.path.join(data_root, '*/*.jpg'))  
    # for i, jpg_file in enumerate(sorted(jpg_files)):
    #     dir_index = i//6250
    #     obj_dir = os.path.join(obj_root, f"{dir_index:06d}")
    #     if not os.path.exists(obj_dir):
    #         os.makedirs(obj_dir)
    #     shutil.move(jpg_file, obj_dir)
    #     # if i >20:
    #     #     break

    blacklist_file = "dataset/exp-data/blacklist/train-4c.txt"
    src_file = "dataset/exp-data/removeredundancy/train.txt"
    pass_file = "output/choose_noise.txt"
    obj_file = "dataset/exp-data/blacklist/train.txt"
    do_write(src_file, blacklist_file, pass_file, obj_file)