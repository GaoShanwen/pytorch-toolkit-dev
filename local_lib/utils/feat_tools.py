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


weights75 = [
    1.8, 1.5, 1.4, 1.36, .967, .86, .75, .74, .63, .55,
    .52, .5, .48, .42, .4, .38, .37, .35, .34, .32,
    .3, .28, .27, .26, .25, .24, .23, .225, .21, .2,
    .15, .145, .125, .11, .097, .095, .092, .09, .087, .084,
    .082, .08, .078, .075, .073, .071, .068, .065, .063, .06,
]
weights65 = [
    1.216, 1.151, 1.059, 1.006, 1.004, .991, .981, .967, .954, .937,
    .923, .91, .892, .878, .861, .845, .814, .792, .77, .761,
    .74, .72, .68, .65, .63, .62, .615, .607, .592, .581,
    .573, .563, .557, .55, .542, .537, .524, .516, .492, .48,
    .46, .454, .441, .438, .421, .42, .4, .38, .37, .35,
]
weights55 = np.array([
    2.22, 2.21, 2.20, 2.19, 2.18, 2.17, 2.16, 2.15, 2.14, 2.13, 
    2.12, 2.11, 2.10, 2.09, 2.08, 2.07, 2.06, 2.05, 2.04, 2.03, 
    2.02, 2.01, 2.00, 1.99, 1.98, 1.97, 1.96, 1.95, 1.94, 1.93, 
    1.92, 1.91, 1.90, 1.89, 1.88, 1.87, 1.86, 1.85, 1.84, 1.83, 
    1.82, 1.81, 1.80, 1.79, 1.78, 1.77, 1.76, 1.75, 1.74, 1.73
])
weights11 = np.ones((50))


def run_compute(p_label, q_label, scores=None, do_output=True, k=5, th=None):
    blacklist = np.arange(110110110001, 110110110010)
    masks = np.any(np.isin(p_label, blacklist), axis=1)
    p_label, q_label = p_label[~masks], q_label[~masks]
    if th is not None and scores is not None:
        scores[:, 0] = 1.
        masks = np.where((scores < th))
        p_label[masks] = np.nan

    top1_num = np.where(p_label[:, 0] == q_label)[0].shape[0]
    top5_num = np.unique(np.where(p_label[:, :k] == q_label[:, np.newaxis])[0]).shape[0]
    display_num = np.count_nonzero(~np.isnan(p_label))
    only_ones = np.sum(np.equal(np.sum(np.isnan(p_label[:, 1:]), axis=1), 4))
    if not do_output:
        return top1_num, top5_num, display_num, only_ones
    print(f"top1-knn(k={k}): {top1_num}/{q_label.shape[0]}|{top1_num/q_label.shape[0]*100:.2f}")
    print(f"top5-knn(k={k}): {top5_num}/{q_label.shape[0]}|{top5_num/q_label.shape[0]*100:.2f}")
    print(f"display-avg(th={th}): {display_num/p_label.shape[0]:.2f}")
    print(f"display-one(th={th}): {only_ones/p_label.shape[0]*100:.2f}")


def compute_acc_by_cat(p_label, q_label, p_scores=None, label_map=None, threshold=None):
    val_dict = {"all_data": q_label.shape[0]}
    val_dict.update(dict(Counter(q_label)))
    acc_map = {}
    for cat, data_num in val_dict.items():
        keeps = np.ones(shape=data_num, dtype=bool) if cat == "all_data" else np.isin(q_label, [cat])
        cat_pl, cat_ql = p_label[keeps], q_label[keeps]
        cat_ps = p_scores[keeps] if p_scores is not None else p_scores
        # cat_pl, cat_ql = p_label.copy(), q_label.copy()
        # cat_ps = p_scores.copy() if p_scores is not None else p_scores
        # if cat != "all_data":
        #     cat_pl[p_label!=cat] = 0
        #     cat_ql[q_label!=cat] = 0
        top1_num, top5_num, display_num, only_ones = \
            run_compute(cat_pl, cat_ql, cat_ps, do_output=False, th=threshold)
        cat_res = {
            "name": label_map[cat] if label_map is not None and cat in label_map else '',
            "gallery_num": 0,
            "query_num": data_num,
            "top1_acc": top1_num / data_num * 100,
            "top5_acc": top5_num / data_num * 100,
            "display_avg": display_num / data_num,
            "display_one": only_ones / data_num * 100
        }
        acc_map.update({cat: cat_res})
    return acc_map


def create_index(data_embedding, use_gpu=False, param="Flat", measure=faiss.METRIC_INNER_PRODUCT):
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
            # 找到大于阈值的得分, 定位到位置
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


def softmax(x, do_sqrt=False):
    if do_sqrt:
        x = np.sqrt(x)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def normalize_scores(scores):
    s = scores.copy()
    s[s<0] = 0
    return s / s.sum(axis=1, keepdims=True)

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
            # 1.8, 1.5, 1.4, 1.36, .967, .86, .75, .74, .63, .55,
            # .52, .5, .48, .42, .4, .38, .37, .35, .34, .32,
            # .3, .28, .27, .26, .25, .24, .23, .225, .21, .2,
            # .15, .145, .125, .11, .097, .095, .092, .09, .087, .084,
            # .082, .08, .078, .075, .073, .071, .068, .065, .063, .06,
            1.216, 1.151, 1.059, 1.006, 1.004, .991, .981, .967, .954, .937,
            .923, .91, .892, .878, .861, .845, .814, .792, .77, .761,
            .74, .72, .68, .65, .63, .62, .615, .607, .592, .581,
            .573, .563, .557, .55, .542, .537, .524, .516, .492, .48,
            .46, .454, .441, .438, .421, .42, .4, .38, .37, .35,
        ]
    )[:choose_pic]
    # weight = np.ones((choose_pic))
    # bias = np.zeros((final_cat,))
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

    _scores = np.dot(input_x, weight) #+ bias
    final_scores = softmax(_scores.reshape(-1, final_cat))
    index_cats = np.array(index_cats)
    I = np.argsort(final_scores, axis=1)[:, ::-1]
    return index_cats[np.arange(index_cats.shape[0])[:, None], I[:, :5]]


def everycat_Npic(scores, plabels, choose_Npic=5, final_cat=5, do_sqrt: bool=True, weights=None):
    final_plabels, final_pscores = [], []
    for score, plabel in zip(scores, plabels):
        final_p, cat_static, index_num = [], {"cats": [], "scores": [], "counts": []}, 0
        for i, (l, s) in enumerate(zip(plabel, score)):
            if l not in cat_static["cats"]:
                cat_static["cats"].append(l)
                cat_static["scores"].append(0.)
                cat_static["counts"].append(0)
            cat_num = cat_static["cats"].index(l)
            if cat_static["counts"][cat_num] > 5:
                continue
            cat_static["counts"][cat_num] += 1
            cat_static["scores"][cat_num] += s * weights[index_num]
            index_num += 1
            if index_num >= len(weights):
                break
        sorted_indexes = sorted(range(len(cat_static["scores"])), key=lambda i: cat_static["scores"][i], reverse=True)
        final_p = np.array(cat_static["cats"])[sorted_indexes].tolist()
        final_s = np.array(cat_static["scores"])[sorted_indexes].tolist()
        if len(final_p) < final_cat:
            final_p = final_p + [np.nan] * (final_cat - len(final_p))
            final_s = final_s + [-1.] * (final_cat - len(final_s))
        final_plabels.append(final_p[:final_cat])
        final_pscores.append(final_s[:final_cat])
    
    return final_plabels, softmax(np.array(final_pscores), do_sqrt)


def merge_topN_scores(scores, plabels, choose_Npic=30, final_cat=5, weights=None):
    assert scores.shape == plabels.shape and scores.shape[1] >= choose_Npic, \
        f"{scores.shape} or {plabels.shape} is error!"
    weights = weights[:choose_Npic]
    final_plabels, final_pscores = [], []
    for score, plabel in zip(scores, plabels):
        final_p, cat_static = [], {"cats": [], "scores": [], "counts": []}
        for i, (l, s) in enumerate(zip(plabel, score)):
            if i >= choose_Npic:
                break
            if l not in cat_static["cats"]:
                cat_static["cats"].append(l)
                cat_static["scores"].append(0.)
                cat_static["counts"].append(0)
            cat_num = cat_static["cats"].index(l)
            cat_static["counts"][cat_num] += 1
            cat_static["scores"][cat_num] += s * weights[i]
        sorted_indexes = sorted(range(len(cat_static["scores"])), key=lambda i: cat_static["scores"][i], reverse=True)
        final_p = np.array(cat_static["cats"])[sorted_indexes].tolist()
        final_s = np.array(cat_static["scores"])[sorted_indexes].tolist()
        if len(final_p) < final_cat:
            final_p = final_p + [np.nan] * (final_cat - len(final_p))
            final_s = final_s + [-1.] * (final_cat - len(final_s))
        final_plabels.append(final_p[:final_cat])
        final_pscores.append(final_s[:final_cat])
    
    return final_plabels, normalize_scores(np.array(final_pscores)) #softmax(np.array(final_pscores))


def get_predict_label(scores, initial_rank, gallery_label, k=5, threshold=None, trick_id=None):
    if trick_id and threshold:
        initial_rank[scores<threshold], scores[scores<threshold] = -1, 0.
    if not trick_id:
        return gallery_label[initial_rank], None
    res = []
    if not trick_id:
        for initial in initial_rank:
            sorted_counter = Counter(gallery_label[initial].tolist()).most_common()
            res += [[counter[0] for counter in sorted_counter[:k]]]
        res = [sublist + [np.nan] * (k - len(sublist)) for sublist in res]
        return np.array(p_labels), None
    if trick_id in [11, 65, 75]:
        weights = eval(f"weights{trick_id}")
        p_labels, p_scores = merge_topN_scores(scores, gallery_label[initial_rank], final_cat=k, weights=weights)
    # elif trick_id == 65:
    #     res = create_matrix(scores, gallery_label[initial_rank], choose_pic=scores.shape[1], final_cat=k)
    elif trick_id == 55:
        p_labels, p_scores = everycat_Npic(scores, gallery_label[initial_rank], final_cat=k, do_sqrt=False, weights=weights)
    return np.array(p_labels), np.array(p_scores)


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
