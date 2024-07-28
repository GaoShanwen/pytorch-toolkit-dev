import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np


class StatisticDistance(object):
    def __init__(self, src_path, dst_dir, keys=["all"]):
        self.src_path = src_path
        self.keys = keys

        for cat in keys:
            self.dst_path = os.path.join(dst_dir, f"{cat}.txt")
            self.get_result(cat)
            self.save_result()

    def statistics_instance(self, obj):
        value = round(math.log10(obj+1e-12), 4)
        if value not in self.distance_dict.keys():
            self.distance_dict[value] = 0
        self.distance_dict[value] += 1

    def get_result(self, cat):
        self.inst_count = 0
        self.distance_dict = {}
        data = np.load(f"{self.src_path}-{cat}.npz")
        pred_list = []
        for score, gt_label in zip(data["scores"], data["labels"]):
            # if cat == "all" or int(gt_label) == self.keys.index(cat):
            pred_list.append(score)

        self.inst_count = len(pred_list)
        for pred in pred_list:
            self.statistics_instance(pred)

    def save_result(self):
        key_sort = list(self.distance_dict.keys())
        key_sort.sort()
        with open(self.dst_path, "w") as save_f:
            line = "instance_number: {} distance_num: {} ".format(self.inst_count, len(self.distance_dict.keys()))
            save_f.writelines(line + "\n")
            inter_num = 0
            for key in key_sort:
                inter_num += self.distance_dict[key]
                rbox_info = "{}".format(self.distance_dict[key] / len(self.distance_dict.keys()))
                line = "{},{},{}".format(float("%.06f" % key), (inter_num / self.inst_count), rbox_info)
                save_f.writelines(line + "\n")
        save_f.close()


class DrawStatisticResult(object):
    def __init__(self, file_dir, keys=["all"]):
        self.attributes = {}
        self.vis_data = {}
        self.keys = keys
        for file_name in self.keys:
            file_path = os.path.join(file_dir, file_name + ".txt")
            self.attributes[file_name] = []
            self.vis_data[file_name] = []
            self.readfile(file_path)
        self.plot_pic(file_dir.split("/")[-1])

    def readfile(self, filename):
        with open(filename, "r") as read_f:
            _ = read_f.readline()
            lines = read_f.readlines()
            splitlines = [x.strip().split(",") for x in lines]
            cat = filename.split("/")[-1].split(".txt")[0]
            for iter, splitline in enumerate(splitlines):
                self.attributes[cat].append(float(splitline[0]))
                self.vis_data[cat].append(float(splitline[1]))

    def plot_pic(self, name, colors=["blue", "green", "orange", "red", "black", "gray"]):
        fig, axes = plt.subplots()
        for iter, key in enumerate(self.keys):
            x = np.array(self.attributes[key])
            y = np.array(self.vis_data[key])
            # 绘制曲线
            plt.plot(x, y, colors[iter], linewidth=2, label=key)

        ax1 = plt.gca()

        # 坐标轴设置
        # ax1.set_title(key[0].split('_')[0])
        min_value = min(v[0] for v in self.attributes.values())
        max_value = max(v[-1] for v in self.attributes.values())
        xticks = [min_value, max_value] + np.arange(min_value, max_value, 0.5).tolist()

        # axes.set_xticks(xticks)
        plt.xticks(xticks, np.around(np.power(10, xticks), decimals=4))
        plt.xticks(rotation=45)
        # dim = (xticks[5]-xticks[0])//5
        # ax1.xaxis.set_ticks(np.arange(xticks[0], xticks[5] +dim, dim))
        plt1_y_min_value, plt1_y_max_value = 0, 1
        axes.set_yticks([])
        ax1.yaxis.set_ticks(np.arange(plt1_y_min_value, plt1_y_max_value + 0.1, 0.1))
        plt.grid(linestyle="--", axis="y")
        plt.figtext(0.9, 0.05, "$X:Score$")
        plt.figtext(0.1, 0.9, "$Y:Ratios$")
        plt.title(f"{name}")
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(loc="upper left")  # , ncol=3

        # it's with question, you can show and save it in plt.
        plt.savefig(f"ratio_{name}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visulize Rank1 Score")
    parser.add_argument("-s", "--src_path", type=str, default="", help="source npz path")
    parser.add_argument("-d", "--dst_path", type=str, default="", help="object txt path")
    args = parser.parse_args()
    StatisticDistance(args.src_path, args.dst_path, keys=["1144-t", "1144-f"])
    DrawStatisticResult(args.dst_path, keys=["1144-t", "1144-f"])
