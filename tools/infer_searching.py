#!/usr/bin/env python3
######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: feat_extract.py
# function: run model to searching imgs and save (train/val).
######################################################
import logging
import os
import shutil
from contextlib import suppress

import cv2
import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import tqdm
from timm.models import load_checkpoint
from timm.utils import setup_default_logging

from local_lib.data.dataset_factory import create_custom_dataset
from local_lib.data.loader import create_custom_loader
from local_lib.models import create_custom_model
from local_lib.utils import parse_args
from local_lib.utils.feat_tools import create_index, get_predict_label
from local_lib.utils.file_tools import init_feats_dir, load_csv_file, load_data, save_dict2csv, save_feat
from local_lib.utils.visualize import run_vis2bigimgs, vis_text

_logger = logging.getLogger("Extract feature")


def load_feats(load_dir="./output/features"):
    files = sorted(os.listdir(load_dir))
    feats = []
    for file_name in files:
        file_path = os.path.join(load_dir, file_name)
        data = np.load(file_path)
        feats.append(data["feats"])
    return np.concatenate(feats)


def load_model(args):
    # might as well try to validate something
    args.pretrained = args.pretrained and not args.checkpoint

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_custom_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=in_chans,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        scriptable=args.torchscript,
        **args.model_kwargs,
    )

    if args.num_classes is None:
        assert hasattr(model, "num_classes"), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, False)
    if "redution" in args.model:
        if "mobilenetv3" in args.model:
            model.classifier = nn.Identity()  # 移除分类层
        elif "regnet" in args.model:
            model.head.fc = nn.Identity()  # 移除分类层
        else:
            raise f"not support {args.model} !"

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info("Model %s created, param count: %d" % (args.model, param_count))

    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))
    return model


def run_infer(model, args):
    device = torch.device(args.device)
    # resolve AMP arguments based on PyTorch / Apex availability
    amp_autocast = suppress

    with open(args.cats_file, "r") as f:
        class_list = [line.strip("\n") for line in f.readlines()]
    assert args.input_mode in ["path", "dir", "file"], "please set infer_mode to path, dir, or files"
    if args.input_mode == "file":
        with open(args.data_path, "r") as f:
            query_files, query_labels = zip(*([line.strip("\n").split(", ") for line in f.readlines()]))
        query_labels = [
            class_list.index(q_label) if q_label in class_list else int(q_label) for q_label in query_labels
        ]
        query_labels = np.array(query_labels, dtype=int)
    else:
        query_files = (
            [os.path.join(args.data_path, path) for path in os.listdir(args.data_path)]
            if args.input_mode == "dir"
            else [args.data_path]
        )
        query_labels = None

    # query_files = np.array(query_files)
    dataset = create_custom_dataset(root=query_files, name="txt_data", split="infer")
    input_size = [3, args.img_size, args.img_size]
    loader = create_custom_loader(
        dataset,
        input_size=input_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        crop_pct=1.0,
        device=device,
        transfrom_mode="custom",
    )

    # data_trans = custom_transfrom()
    pbar = tqdm.tqdm(total=len(loader))
    model.eval()
    init_feats_dir(args.results_dir)
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            if device == "cuda":
                input = input.to(device)

            with amp_autocast():
                output = model(input)
            save_feat(output.cpu().numpy(), batch_idx, args.results_dir)
            pbar.update(1)
    pbar.close()
    return query_labels, query_files


def load_names(label_file):
    product_id_map = {}
    with open(label_file, "r") as f:
        for line in f.readlines()[1:]:
            try:
                id_record = line.strip().replace('"', "").split(",")
                product_id_map.update({int(id_record[1]): id_record[2]})
            except:
                print(f"line={line} is error!")
    return product_id_map


def static_search_res(I, g_labels, q_labels):
    from collections import Counter

    cats = list(set(q_labels))
    label_index = load_csv_file(args.label_file)
    label_map = {int(cat): name.split("/")[0] for cat, name in label_index.items()}
    q_maps = load_names("dataset/function_test/1029.csv")
    search_res = {}
    for cat in cats:
        keeps = np.where(q_labels == cat)[0]
        choose_idx = np.unique(I[keeps].reshape(-1))
        index_res = g_labels[choose_idx].tolist()
        first_count = Counter(index_res).most_common()
        search_num = len(index_res)
        cat_static = {"query_num": keeps.shape[0], "index": cat}
        for i, (search_cat, count) in enumerate(first_count[:5]):
            cat_static.update(
                {
                    f"top{i+1}_name": label_map[search_cat] if search_cat in label_map else search_cat,
                    f"top{i+1}_ratio": count / search_num,
                }
            )
        search_res.update({q_maps[cat]: cat_static})
    return search_res


def copy_error(I, g_labels, q_labels, g_files, save_root="output/temp"):
    from collections import Counter

    choose_dict = {
        "沙地德源腊鸭腿": ["猪头肉"],
    }
    cats = list(set(q_labels))
    label_index = load_csv_file(args.label_file)
    label_map = {int(cat): name.split("/")[0] for cat, name in label_index.items()}
    q_maps = load_names("dataset/function_test/1029.csv")
    for cat in cats:
        if q_maps[cat] not in choose_dict.keys():
            continue
        keeps = np.where(q_labels == cat)[0]
        choose_idx = np.unique(I[keeps].reshape(-1))
        index_res = g_labels[choose_idx].tolist()
        first_count = Counter(index_res).most_common()
        for searched_cat, _ in first_count[:5]:
            if label_map[searched_cat] not in choose_dict[q_maps[cat]]:
                continue
            need_index = np.where(g_labels[I[keeps]] == searched_cat)  # [0]
            need_index = I[keeps][need_index]
            # import pdb; pdb.set_trace()
            save_dir = os.path.join(save_root, label_map[searched_cat])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for file_path in g_files[need_index]:
                if not os.path.exists(file_path):
                    continue
                shutil.copy(file_path, save_dir)


def static_top1(top1_scores, categories, save_root):
    np.savez(os.path.join(save_root, "static.npz"), scores=top1_scores, labels=categories)


def run_search(q_labels, query_files, args):
    # args.param = "Flat"
    args.param = f"IVF629,Flat"
    args.measure = faiss.METRIC_INNER_PRODUCT
    # 加载npz文件
    g_feats, g_labels, g_files = load_data(args.gallerys)
    # gallery_feature = gallery_feature[3765:]
    query_feats = load_feats(args.results_dir)
    faiss.normalize_L2(g_feats)
    faiss.normalize_L2(query_feats)

    # base_name = os.path.basename(args.data_path)
    # if base_name[:-4] in ["search", "1029"]:
    #     searched_data = np.load("output/feats/searched_res-148c.npy")
    #     choose_idx = np.arange(g_feats.shape[0])
    #     g_feats = g_feats[choose_idx]
    index = create_index(g_feats, use_gpu=args.use_gpu, param=args.param, measure=args.measure)
    D, I = index.search(query_feats, args.topk)

    # static_res = static_search_res(I, g_labels, q_labels)
    # save_dict2csv(static_res, "1029_static.csv")
    # copy_error(I, g_labels, q_labels, g_files)
    static_top1(D[:, 0], q_labels, args.save_root)
    return
    # import pdb; pdb.set_trace()
    # if base_name[:-4] in ["search", "1029"]:
    #     res = np.concatenate((q_labels[:, np.newaxis], I), axis=1)
    #     # np.save('output/feats/searched_res-148c.npy', res)
    #     np.save('output/feats/searched_res-1029.npy', res)
    #     return

    # search_res = {data[0]: data[1:].tolist() for data in searched_data}
    # tp_nums = 0
    # for q_l, g_idx in zip(q_labels, I):
    #     g_searhed = search_res[q_l]
    #     pred_tp = np.in1d(choose_idx[g_idx], g_searhed)
    #     if pred_tp.any():
    #         tp_nums += 1
    # print(f"{tp_nums} / {q_labels.shape[0]}")

    # blacklist = np.arange(110110110001, 110110110010)
    blacklist = np.array([110110110001, 110110110002] + list(range(110110110005, 110110110010)))
    # import pdb; pdb.set_trace()
    black_nums = np.sum(np.isin(g_labels[I][:, :11], blacklist), axis=1)
    choose_idx = np.where(black_nums >= 2)[0]

    # masks = np.any(np.isin(g_labels[I][:, :11], blacklist), axis=1)
    # choose_idx = np.where(masks == True)[0]

    for i in choose_idx:
        # shutil.copy(query_files[i], args.save_root)
        sum_img = np.full((640, 2408, 3), 255, dtype=np.uint8)  # 生成全白大图
        img = cv2.imread(query_files[i])
        H, W, _ = img.shape
        new_height = min(640, int(H * (480 / W)))  # 保持原来的长宽比
        img = cv2.resize(img, (480, new_height))  # 调整大小
        color = (0, 255, 0)
        img = vis_text(img, str(q_labels[i]), [2, 2], color, text_size=48)
        sum_img[0:new_height, 0:480, :] = img
        start_w = 482
        for j, cat_idx in enumerate(g_labels[I][i]):
            if cat_idx not in blacklist:
                continue
            color = (255, 0, 0)
            img = cv2.imread(g_files[I[i, j]])
            H, W, _ = img.shape
            new_height = min(640, int(H * (480 / W)))  # 保持原来的长宽比
            img = cv2.resize(img, (480, new_height))  # 调整大小
            img = vis_text(img, str(cat_idx), [2, 2], color, text_size=48)
            img = vis_text(img, str(round(D[i, j], 3)), [2, new_height - 48 - 2], color, text_size=48)
            sum_img[0 : 0 + new_height, start_w : start_w + 480, :] = img
            start_w += 482
            if start_w >= sum_img.shape[1]:
                break
        cv2.imwrite(os.path.join(args.save_root, query_files[i].split("/")[-1]), sum_img)

    # p_labels = gallery_labels[I]
    # cats = list(set(gallery_labels))
    # label_index = load_csv_file(args.label_file)
    # label_map = {i: label_index[cat] for i, cat in enumerate(class_list) if i in cats}
    # q_labels = q_labels if q_labels is not None else p_labels[:, 0]
    # run_vis2bigimgs(I, gallery_labels, gallery_files, q_labels, query_files, label_map, args.save_root)


if __name__ == "__main__":
    setup_default_logging()
    args, _ = parse_args()
    model = load_model(args)
    q_labels, q_files = run_infer(model, args)
    run_search(q_labels, q_files, args)
