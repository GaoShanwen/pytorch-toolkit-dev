######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.06.14
# filenaem: vis_recognize_error.py
# function: visualize the error picture for recognize results.
######################################################
import numpy as np
import os
import shutil
import glob
from local_lib.utils.visualize import VisualizeResults


def save2txt(obj_root, cats, file_path="expend.txt"):
    with open(file_path, "w") as f:
        for cat in os.listdir(obj_root):
            obj_dir = os.path.join(obj_root, cat)
            if not os.path.exists(obj_dir):
                continue
            for file in os.listdir(obj_dir):
                if not file.endswith(".jpg"):
                    continue
                f.write(f"{os.path.join(obj_dir, file)},{cats.index(cat):08d}\n")


if __name__ == '__main__':
    # set the labels for the dataset
    # class_map = {0: "usual", 1: "bag", 2: "bandf", 3: "box", 4: "film", 5: "web", 6: "brand", 7: "tape"}
    cats = ["usual", "bag", "bandf", "box", "film", "net", "brand", "tape"]
    # set the path of the error picture
    recognize_result_path = "recognize_scores.npz"
    # load the recognize results
    loaded_results = np.load(recognize_result_path)
    # get the error picture index
    pscores, gts, filepaths = loaded_results["pscores"], loaded_results["gts"], loaded_results["files"]
    print(f"load results number={gts.shape[0]}")
    high_th, low_th = 0.7, 0.2
    choises_idx = np.argmax(pscores[:, 4:], axis=1) + 4
    # high_th, low_th = 0.9, 0.0
    # choises_idx = np.argmax(pscores, axis=1)
    choises = ((np.arange(gts.shape[0]), choises_idx))
    keeps = np.where((pscores[choises] <= high_th) & (pscores[choises] >= low_th))[0]
    print(f"choices {keeps.shape[0]} imgs")
    # get the error picture
    pscores, filepaths, choises_idx = pscores[choises][keeps], filepaths[keeps], choises_idx[keeps]
    # # visualize the error picture
    # for cat in ["bandf", "film"]:
    #     cats.remove(cat) # remove the unneed categories
    # class_map = {i: cat for i, cat in enumerate(cats)}
    # save_root, text_size = "output/vis/errors", 48 # args.save_root, args.text_size
    # visualizer = VisualizeResults(save_root, "classify", text_size=text_size, class_map=class_map)
    # visualizer.do_visualize(choises_idx, filepaths, choises_idx[:, np.newaxis], scores=pscores[:, np.newaxis])
    
    obj_root = "dataset/function_test/package_way/expend2"
    check_root = "dataset/function_test/package_way/val-exp"
    original_names = [os.path.basename(file) for file in filepaths]
    for cat in cats:
        check_dir = os.path.join(check_root, cat)
        if not os.path.exists(check_dir):
            continue
        expend_names = np.array(list(os.listdir(check_dir)))
        keeps = np.isin(original_names, expend_names)
        obj_dir = os.path.join(obj_root, cat)
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)
        print(f"cat={cat} choices {np.sum(keeps)} imgs")
        for file in filepaths[keeps]:
            shutil.copy(file, obj_dir)
    save2txt(obj_root, cats)
    
