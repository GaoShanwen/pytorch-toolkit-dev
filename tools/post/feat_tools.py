######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: feat_tools.py
# function: the functions tools for feat extract or eval.
######################################################
import tqdm
import faiss
import numpy as np
import pandas as pd
from collections import Counter


def load_data(file_path):
    with np.load(file_path) as data:
        # 从npz文件中获取数组
        feas = data["feats"].astype("float32")
        labels = data["gts"]
        fpaths = data["fpaths"]
    return feas, labels, fpaths


def run_compute(p_label, q_label, do_output=True, k=5):
    tp1_num = np.where(p_label[:, 0] == q_label)[0].shape[0]
    tp5_num = np.unique(np.where(p_label[:, :k] == q_label[:, np.newaxis])[0]).shape[0]
    if do_output:
        return tp1_num, tp5_num
    print(f"top1-knn(k={k}): {tp1_num}/{q_label.shape[0]}|{tp1_num/q_label.shape[0]}")
    print(f"top5-knn(k={k}): {tp5_num}/{q_label.shape[0]}|{tp5_num/q_label.shape[0]}")


def compute_acc_by_cat(p_label, q_label, class_list, label_map=None):
    top1_num, top5_num = run_compute(p_label, q_label)
    acc_map = {
        "all_data": {
            "cat_index": "",
            "top1_num": top1_num,
            "top1_acc": top1_num / q_label.shape[0],
            "top5_num": top5_num,
            "top5_acc": top5_num / q_label.shape[0],
            "data_num": q_label.shape[0],
        }
    }
    val_dict = dict(Counter(q_label))
    for cat, data_num in val_dict.items():
        cat_index = np.where(q_label == cat)
        top1_num = np.where(p_label[cat_index, 0] == q_label[cat_index])[1].shape[0]
        top5_num = np.unique(np.where(p_label[cat_index, :] == q_label[cat_index, np.newaxis])[1]).shape[0]
        cat_res = {
            "cat_index": cat,
            "top1_num": top1_num,
            "top1_acc": top1_num / data_num,
            "top5_num": top5_num,
            "top5_acc": top5_num / data_num,
            "data_num": data_num,
        }
        if label_map is not None:
            cat_res.update({"name": label_map[cat]})
        acc_map[class_list[cat]] = cat_res
    return acc_map


def print_acc_map(acc_map, csv_name):
    df = pd.DataFrame(acc_map).transpose()
    df.to_csv(csv_name)
    # print(df)


def save_keeps_file(labels, files, class_list, obj_files):
    with open(obj_files, "w") as f:
        for label_index, filename in zip(labels, files):
            label = class_list[label_index]
            f.write(f"{filename}, {label}\n")


def create_index(data_embedding, use_gpu=False, param="Flat", measure=faiss.METRIC_INNER_PRODUCT, L2_flag=False):
    dim = data_embedding.shape[1]
    index = faiss.index_factory(dim, param, measure)
    if param.startswith("IVF"):
        faiss.ParameterSpace().set_index_parameters(index, "nprobe=3")
    if use_gpu:
        index = faiss.index_cpu_to_gpus_list(index, gpus=[0, 1])  # gpus用于指定使用的gpu号
    index.train(data_embedding)
    index.add(data_embedding)  # 把向量数据加入索引
    return index


def choose_similarity(matrix, labels, samilar_thresh=0.9, use_gpu=False, update_times=0):
    cats = list(set(labels))
    stride = len(cats) // update_times if update_times else 1
    keeps = []
    pbar = tqdm.tqdm(total=len(cats), miniters=stride, maxinterval=3600)
    for cat in cats:
        cat_index = np.where(labels == cat)[0]
        choose_gallery = matrix[cat_index]
        enable_gpu = use_gpu if cat_index.shape[0] <= 6000 else False
        index = create_index(choose_gallery, enable_gpu)
        final_num = min(cat_index.shape[0], 2048) if cat_index.shape[0] <= 6000 else 4096
        # if cat_index.shape[0] <= 15000:
        #     final_num = 6144
        g_g_scores, g_g_indexs = index.search(choose_gallery, final_num)
        # if max(g_g_scores[:, final_num-1])>=0.9:
        #     print(cat, cat_index.shape[0], max(g_g_scores[:, final_num-1]))
        masks = []
        for i in range(cat_index.shape[0] - 1):
            if i in masks:
                continue
            # 找到大于阈值的得分, 定位到位置，
            mask_index = np.where((g_g_scores[i, :] >= samilar_thresh) & (g_g_indexs[i, :] > i))[0]
            masks += g_g_indexs[i, mask_index].tolist()
        masks = np.array(masks)
        keeps += np.setdiff1d(cat_index, cat_index[masks]).tolist() if masks.shape[0] else cat_index.tolist()
        pbar.update(1)
    pbar.close()
    return np.array(keeps)


def get_predict_label(initial_rank, gallery_label, k=5, use_knn=False, use_sgd=False):
    if not use_knn:
        return gallery_label[initial_rank]
    res = []
    if not use_sgd:
        for initial in initial_rank:
            sorted_counter = Counter(gallery_label[initial].tolist()).most_common()
            res += [[counter[0] for counter in sorted_counter[:k]]]
        res = [sublist + [np.nan] * (k - len(sublist)) for sublist in res]
        return np.array(res)
    raise "not support sgd yet!"


def choose_noise(matrix, labels, choose_ratio, use_gpu=False, update_times=0):
    param, measure = "Flat", faiss.METRIC_L2
    keeps = []
    cats = list(set(labels))
    stride = len(cats) // update_times if update_times else 1
    pbar = tqdm.tqdm(total=len(cats), miniters=stride, maxinterval=600)
    for cat in cats:
        cat_index = np.where(labels == cat)[0]
        choose_gallery, gallery_num = matrix[cat_index], cat_index.shape[0]
        index = create_index(choose_gallery, use_gpu, param, measure)
        choose_num = int(gallery_num * choose_ratio)
        final_num = min(choose_num, 2048)
        _, index_matric = index.search(choose_gallery, final_num)
        last_count = Counter(index_matric.reshape(-1).tolist()).most_common()
        keep = [key for key, _ in last_count[:choose_num]]
        keeps += cat_index[np.array(keep, dtype=int)].tolist()
        pbar.update(1)
    pbar.close()
    return np.array(keeps)


def choose_with_static(matrix, labels, _, use_gpu=False, update_times=0):
    all_indics = np.arange(labels.shape[0])
    cats = list(set(labels))
    search_res = {}
    # other_gallery = matrix[others]
    others = all_indics
    index = create_index(matrix, use_gpu)
    stride = len(cats) // update_times if update_times else 1
    pbar = tqdm.tqdm(total=len(cats), miniters=stride, maxinterval=600)
    for cat in cats:
        keeps = np.where(labels == cat)[0]
        # in_set = np.in1d(all_indics, keeps)
        # others = all_indics[~in_set]
        # other_gallery = matrix[others]
        # enable_gpu = use_gpu #if choose_gallery.shape[0] <= 2048 else False
        # index = create_index(other_gallery, enable_gpu)

        choose_gallery = matrix[keeps]
        choose_num = 30  # min(40, int(keeps.shape[0] * choose_ratio))
        _, index_matric = index.search(choose_gallery, choose_num + 1)
        index_res = labels[others[index_matric[:, 1:].reshape(-1)]].tolist()
        first_count = Counter(index_res).most_common()
        search_num = len(index_res)
        cat_static = {"gallery_num": others.shape[0], "query_num": keeps.shape[0]}
        for i, (search_cat, count) in enumerate(first_count[:5]):
            cat_static.update(
                {
                    f"top{i+1}_name": search_cat,
                    f"top{i+1}_error": search_num - count,
                    f"top{i+1}_num": count,
                    f"top{i+1}_ratio": count / search_num,
                }
            )
        search_res.update({cat: cat_static})
        pbar.update(1)
    pbar.close()
    return search_res


def run_choose(matrix, labels, args):
    threshold = args.threshold
    use_gpu = args.use_gpu
    _times = args.update_times
    remove_mode = args.remove_mode
    assert remove_mode in ["noise", "similarity", "static"], f"{remove_mode} is not support yet!"
    function_name = "choose_with_static" if remove_mode == "static" else f"choose_{remove_mode}"
    return eval(function_name)(matrix, labels, threshold, use_gpu, _times)
