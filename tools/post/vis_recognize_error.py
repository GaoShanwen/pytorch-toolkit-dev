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
    if False:
    # if True:
        # # high_th, low_th = 0.5, 0.1
        # # choises_idx = np.argmax(pscores[:, 4:], axis=1) + 4
        # high_th, low_th = 0.99, 0.0
        # choises_idx = np.argsort(-pscores[:, 1:], axis=1)
        # choises = ((np.arange(gts.shape[0]), choises_idx[:, 0]))
        # keeps = np.ones(gts.shape[0]).astype(bool)#np.where((pscores[choises] <= high_th) & (pscores[choises] >= low_th))[0]

        gts = np.array([int(gt)-2 if int(gt)>2 else int(gt) for gt in gts])
        choises_idx = np.argmax(pscores, axis=1)
        choises = ((np.arange(gts.shape[0]), choises_idx))
        keeps = np.where((choises_idx != gts) & (gts != 6))[0]

        # get the error picture
        pscores, choice_files, choises_idx = pscores[choises][keeps], filepaths[keeps], choises_idx[keeps] #
        print(f"choices {choises_idx.shape[0]} imgs")
    # if False:
    if True:
        choices = np.where((pscores[:, 0] <= 0.8))[0]
        masks = np.where((pscores[:, 4] <= 0.2) & (pscores[:, 5] <= 0.2))[0]
        choices = choices[~np.isin(choices, masks)]
        # pscores, choice_files, choises_idx = pscores[choices], filepaths[choices], choises_idx[choices]
        # choices = np.where((pscores[:, 0] <= 0.9) & (pscores[:, 0] >= 0.3))[0]
        pscores, choice_files, choice_gts = pscores[choices], filepaths[choices], gts[choices]
        choises_idx = np.argmax(pscores[:, 1:], axis=1) + 1
        choises = ((np.arange(choice_gts.shape[0]), choises_idx))
        # keeps = np.isin(choises_idx, [1,2,3,4,5,])
        keeps = np.isin(choises_idx, [4,5,])
        
        choises_idx, choice_files, choice_gts, pscores = choises_idx[keeps], choice_files[keeps], choice_gts[keeps], pscores[choises][keeps]

        masks = np.array([filepath.startswith("dataset/optimize_task3/expend/") for filepath in choice_files])
        pscores, choice_files, choises_idx, choice_gts = pscores[~masks], choice_files[~masks], choises_idx[~masks], choice_gts[~masks]
        print(f"choices {choises_idx.shape[0]} imgs")
        # final_masks = np.zeros(gts.shape[0], dtype=bool)
        # final_masks[choices[keeps][~masks]] = True
        # gts, filepaths = gts[~final_masks], filepaths[~final_masks]
        # with open("train.txt", "w") as f:
        #     for filepath, gt in zip(filepaths, gts):
        #         f.write(f"{filepath},{gt}\n")
        #     for filepath, gt, pred in zip(choice_files, choice_gts, choises_idx):
        #         if gt.startswith("110444400"):
        #             gt = gt[10:]
        #         gt = f"1104444{pred:03d}{gt}"
        #         f.write(f"{filepath},{gt}\n")
    # visualize the error picture
    for cat in ["bandf", "film"]:
        cats.remove(cat) # remove the unneed categories
    class_map = {i: cat for i, cat in enumerate(cats)}
    save_root, text_size = "output/vis/errors", 36 # args.save_root, args.text_size
    visualizer = VisualizeResults(save_root, "classify", text_size=text_size, class_map=class_map)
    visualizer.do_visualize(choises_idx, choice_files, choises_idx[:, np.newaxis], scores=pscores[:, np.newaxis])
    # visualizer.do_visualize(choises_idx[:, 0]+1, choice_files, np.array([[1,2,3,4,5]] * gts.shape[0]), scores=pscores[:, 1:])

    # with open("expend-brand.txt", "r") as f:
    #     err_fpaths, err_gts = zip(*([line.strip().split(",") for line in f.readlines()]))
    # err_basepath, err_gts = np.array([os.path.basename(path) for path in err_fpaths]), np.array(err_gts)

    # with open("dataset/function_test/package_way/trainv1.txt", "r") as f:
    #     fpaths, gts = zip(*([line.strip().split(",") for line in f.readlines()]))
    # fpaths, basepath, gts = np.array(fpaths), np.array([os.path.basename(path) for path in fpaths]), np.array(gts)
    # # print(f"choices {choices.shape[0]} imgs")
    # with open("final_trainv1.txt", "w") as f:
    #     for i in range(8):
    #         err_keeps = np.isin(err_gts, [f"{i:08d}"])
    #         choices_errs = err_basepath[err_keeps] 
    #         choices = np.isin(basepath, choices_errs)
    #         if not sum(choices):
    #             continue
    #         print(f"cat={i} {cats[i]} choices {np.sum(choices)} imgs")
    #         final_fpaths = np.array(fpaths)[choices]
    #         for filepath in final_fpaths:
    #             f.writelines(f"{filepath},{i:08d}\n")
    #     others = ~np.isin(basepath, err_basepath)
    #     fpaths, gts = fpaths[others], gts[others]
    #     for filepath, gt in zip(fpaths, gts):
    #         f.writelines(f"{filepath},{gt}\n")

    # obj_root = "dataset/function_test/package_way/expend2"
    # check_root = "dataset/function_test/package_way/val-exp"
    # original_names = [os.path.basename(file) for file in choice_files]
    # for cat in cats:
    #     check_dir = os.path.join(check_root, cat)
    #     if not os.path.exists(check_dir):
    #         continue
    #     expend_names = np.array(list(os.listdir(check_dir)))
    #     keeps = np.isin(original_names, expend_names)
    #     obj_dir = os.path.join(obj_root, cat)
    #     if not os.path.exists(obj_dir):
    #         os.makedirs(obj_dir)
    #     print(f"cat={cat} choices {np.sum(keeps)} imgs")
    #     for file in choice_files[keeps]:
    #         shutil.copy(file, obj_dir)
    
    # # save2txt(obj_root, cats)
    # masks_ = np.zeros(gts.shape[0], dtype=bool)
    # masks_[choices[keeps][~masks]] = True
    # gts, filepaths = gts[~masks_], filepaths[~masks_]
    # with open("expend-brand.txt", "w") as f:
    #     for filepath, gt, pred in zip(choice_files, choice_gts, choises_idx):
    #         if gt.startswith("110444400"):
    #             gt = gt[10:]
    #         gt = f"1104444{pred:03d}{gt}"
    #         f.write(f"{filepath},{gt}\n")
    #     for filepath, gt in zip(filepaths, gts):
    #         f.write(f"{filepath},{gt}\n")
