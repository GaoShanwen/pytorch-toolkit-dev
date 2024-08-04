######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.08.04
# filenaem: get_samilar_cats.py
# function: get the similar categories of the top 2-5 categories.
######################################################
import numpy as np


def return_samilirity_cats(static_v, th):
    return {
        static_v[f"top{idx}_name"]: static_v[f"top{idx}_ratio"]
        for idx in range(2, 6)
        if static_v[f"top{idx}_ratio"] >= th
    }


def print_static(static_res, th=0.01):
    cats = list(static_res.keys())
    masks = np.logical_not(np.ones(len(static_res)))

    for i, (k, v) in enumerate(static_res.items()):
        if masks[i]:
            continue
        check_objs = list(return_samilirity_cats(v, th).items())
        print(f"{k}: {check_objs}")
        masks[i] = True
        while len(check_objs):
            obj = check_objs.pop(0)[0]
            idx = cats.index(obj)
            if masks[idx]:
                continue
            searched = list(return_samilirity_cats(static_res[obj], th).items())
            print(f"{obj}: {searched}")
            check_objs += searched
            masks[idx] = True
        print()
