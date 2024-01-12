######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: feat_tools.py
# function: the functions tools for feat extract or eval.
######################################################
import math
from collections import Counter

import faiss
import numpy as np
import tqdm


def run_compute(p_label, q_label, do_output=True, k=5):
    blacklist = np.arange(110110110001, 110110110010)
    masks = np.any(np.isin(p_label, blacklist), axis=1)
    p_label, q_label = p_label[~masks], q_label[~masks]

    tp1_num = np.where(p_label[:, 0] == q_label)[0].shape[0]
    tp5_num = np.unique(np.where(p_label[:, :k] == q_label[:, np.newaxis])[0]).shape[0]
    if do_output:
        return tp1_num, tp5_num
    print(f"top1-knn(k={k}): {tp1_num}/{q_label.shape[0]}|{tp1_num/q_label.shape[0]}")
    print(f"top5-knn(k={k}): {tp5_num}/{q_label.shape[0]}|{tp5_num/q_label.shape[0]}")


def compute_acc_by_cat(p_label, q_label, label_map=None):
    top1_num, top5_num = run_compute(p_label, q_label)
    acc_map = {
        "all_data": {
            "top1_num": top1_num,
            "top1_acc": top1_num / q_label.shape[0],
            "top5_num": top5_num,
            "top5_acc": top5_num / q_label.shape[0],
            "data_num": q_label.shape[0],
            "name": "",
        }
    }
    val_dict = dict(Counter(q_label))
    for cat, data_num in val_dict.items():
        cat_index = np.where(q_label == cat)
        top1_num = np.where(p_label[cat_index, 0] == q_label[cat_index])[1].shape[0]
        top5_num = np.unique(np.where(p_label[cat_index, :] == q_label[cat_index, np.newaxis])[1]).shape[0]
        cat_res = {
            "top1_num": top1_num,
            "top1_acc": top1_num / data_num,
            "top5_num": top5_num,
            "top5_acc": top5_num / data_num,
            "data_num": data_num,
        }
        if label_map is not None:
            cat_res.update({"name": label_map[cat]})
        acc_map[cat] = cat_res
    return acc_map


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


def get_thresh(samilar_thresh, data_num):
    if data_num <= 50:
        return 1.
    if isinstance(samilar_thresh, float):
        return samilar_thresh
    for v, set_th in samilar_thresh:
        if data_num <= v:
            break
    return set_th


def intra_similarity(matrix, labels, samilar_thresh=0.9, use_gpu=False, update_times=0):
    cats = list(set(labels))
    stride = len(cats) // update_times if update_times else 1
    masks = []
    pbar = tqdm.tqdm(total=len(cats), miniters=stride, maxinterval=3600)
    for cat in cats:
        pbar.update(1)
        cat_index = np.where(labels == cat)[0]
        data_num = cat_index.shape[0]
        threshold = get_thresh(samilar_thresh, data_num)
        if abs(threshold - 1.) < 1e-9:
            continue
        choose_gallery = matrix[cat_index]
        index = create_index(choose_gallery, use_gpu)
        g_g_scores, g_g_indexs = index.search(choose_gallery, min(data_num, 2048))
        mask = np.logical_not(np.ones((data_num)))
        for i in range(data_num - 1):
            if mask[i]:
                continue
            # 找到大于阈值的得分, 定位到位置，
            mask_index = np.where((g_g_scores[i, :] >= threshold) & (g_g_indexs[i, :] > i))[0]
            mask[g_g_indexs[i, mask_index]] = True
        masks += cat_index[np.where(mask == True)[0]].tolist()
    pbar.close()
    return np.setdiff1d(np.arange(labels.shape[0]), masks)


def inter_similarity(matrix, labels, samilar_thresh=0.9, use_gpu=False, update_times=0):
    split_times = 10
    data_num = labels.shape[0]
    step = math.ceil(data_num / split_times)
    masks = np.logical_not(np.ones((data_num)))
    index = create_index(matrix, use_gpu)
    pbar = tqdm.tqdm(total=data_num)
    for i in range(split_times):
        search_feats = matrix[i * step : (i + 1) * step]
        g_g_scores, g_g_indexs = index.search(search_feats, 2048)
        for j in range(g_g_indexs.shape[0] - 1):
            pbar.update(1)
            pos = i * step + j
            if masks[pos]:
                continue
            # 找到大于阈值的得分, 定位到位置
            mask_index = np.where((g_g_scores[j, :] >= samilar_thresh) & (g_g_indexs[j, :] > pos))[0]
            masks[g_g_indexs[j, mask_index]] = True
    pbar.close()
    return np.where(masks == False)[0]


def create_matrix(scores, plabels, choose_pic=30, final_cat=5):
    """_summary_

    Args:
        scores (np.array): the scores topk(pictures-level)
        plabels (_type_): the categories topk(pictures-level)
        choose_pic (int, optional): input topk for pictures. Defaults to 30.
        final_cat (int, optional): output topk for categories. Defaults to 5.

    Returns:
        index_res: the categories topk(categories-level)
    """
    weight = np.array(
        [
            1.8, 1.5, 1.4, 1.36, 0.967, 0.86, 0.75, 0.74,
            0.63, 0.55, 0.52, 0.5, 0.48, 0.42, 0.4, 0.38, 
            0.37, 0.35, 0.34, 0.32, 0.3, 0.28, 0.27, 0.26, 
            0.25, 0.24, 0.23, 0.225, 0.21, 0.2,
            .15, .145, .125, .11, .097, .095, .092, .09, .087, .084,
            .082, .08, .078, .075, .073, .071, .068, .065, .063, .06,
        ]
    )[:choose_pic]
    # weight = np.ones((choose_pic))
    bias = np.zeros((final_cat,))
    input_x, index_cats = [], []
    
    for score, plabel in zip(scores, plabels):
        final_l = Counter(plabel.tolist()).most_common()[:final_cat]
        final_l = [l for l, _ in final_l] + [np.nan] * (final_cat - len(final_l))
        input = np.zeros((final_cat, choose_pic))
        for i, (l, s) in enumerate(zip(plabel, score)):
            if l not in final_l:
                continue
            input[final_l.index(l), i] = s
        input_x.append(input)
        index_cats.append(final_l)

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    _scores = np.dot(input_x, weight) + bias
    final_scores = softmax(_scores.reshape(-1, final_cat))
    index_cats = np.array(index_cats)
    I = np.argsort(final_scores, axis=1)[:, ::-1]
    return index_cats[np.arange(index_cats.shape[0])[:, None], I[:, :5]]


def get_predict_label(scores, initial_rank, gallery_label, k=5, use_knn=False, weighted=False):
    if not use_knn:
        return gallery_label[initial_rank]
    res = []
    if not weighted:
        for initial in initial_rank:
            sorted_counter = Counter(gallery_label[initial].tolist()).most_common()
            res += [[counter[0] for counter in sorted_counter[:k]]]
        res = [sublist + [np.nan] * (k - len(sublist)) for sublist in res]
        return np.array(res)
    res = create_matrix(scores, gallery_label[initial_rank], choose_pic=scores.shape[1], final_cat=k)
    return np.array(res)


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


def choose_with_static(matrix, labels, _, use_gpu=False, update_times=0, choose_num=30):
    all_indics = np.arange(labels.shape[0])
    cats = list(set(labels))
    search_res = {}
    index = create_index(matrix, use_gpu)
    stride = len(cats) // update_times if update_times else 1
    pbar = tqdm.tqdm(total=len(cats), miniters=stride, maxinterval=600)
    for cat in cats:
        keeps = np.where(labels == cat)[0]
        choose_gallery = matrix[keeps]

        _, index_matric = index.search(choose_gallery, choose_num + 1)
        index_res = labels[all_indics[index_matric[:, 1:].reshape(-1)]].tolist()
        first_count = Counter(index_res).most_common()
        search_num = len(index_res)
        cat_static = {"query_num": keeps.shape[0]}
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
    assert remove_mode in [
        "choose_noise",
        "intra_similarity",
        "inter_similarity",
        "choose_with_static",
    ], f"{remove_mode} is not support yet!"
    return eval(remove_mode)(matrix, labels, threshold, use_gpu, _times)
