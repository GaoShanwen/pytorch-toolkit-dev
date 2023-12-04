######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: meter.py
# function: eval accuracy of every classes.
######################################################
import torch
import collections
import pandas as pd


class ClassAccuracyMap:
    """Computes and stores the accuracy current value"""

    def __init__(self, label_map):
        self.label_map = {value: key for key, value in label_map.items()}
        self.reset()

    def reset(self):
        self.obj_dict = {}

    def update_cat(self, cat, top1, top5, num):
        name = self.label_map[cat] if cat != "all" else cat
        if name not in self.obj_dict.keys():
            key_names = ["top1_num", "top1_val", "top5_num", "top5_val", "count"]
            cat_dict = {key_name: 0 for key_name in key_names}
        else:
            cat_dict = self.obj_dict[name]
        count = cat_dict["count"] + num
        top1_num = cat_dict["top1_num"] + top1
        top5_num = cat_dict["top5_num"] + top5
        cat_dict["top1_num"] = top1_num
        cat_dict["top5_num"] = top5_num
        cat_dict["top1_val"] = top1_num * 100.0 / count
        cat_dict["top5_val"] = top5_num * 100.0 / count
        cat_dict["count"] = count
        self.obj_dict[name] = cat_dict

    def update(self, output, target, topk=(1, 5)):
        maxk = min(5, output.size()[1])
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred)).to("cpu")
        target = target.to("cpu")
        counter = collections.Counter(target.tolist())
        for cat, num in counter.items():
            indices = torch.where(target == cat)[0]
            top1, top5 = [
                correct[: min(k, maxk), indices].reshape(-1).float().sum(0)
                for k in topk
            ]
            self.update_cat(cat, top1.item(), top5.item(), num)
        top1, top5 = [correct[: min(k, maxk)].reshape(-1).float().sum(0) for k in topk]
        self.update_cat("all", top1.item(), top5.item(), batch_size)

    def save_to_csv(self, task="val"):
        df = pd.DataFrame(self.obj_dict).transpose()
        df.to_csv(f"eval_res-{task}.csv")
