import umap
import faiss
import numpy as np
import os
import shutil
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # 用于自定义图例项

from local_lib.utils.file_tools import load_data, load_csv_file
from tqdm import tqdm

import seaborn as sns

from mplfonts import use_font
use_font('Noto Serif CJK SC')
# use_font('Noto Sans CJK SC')#指定中文字体

def save_to_html(save_name, X_umap_2d, g_label, g_choics, width=1500, height=1200):
    df_2d = pd.DataFrame()
    df_2d['X'] = list(X_umap_2d[:, 0].squeeze())
    df_2d['Y'] = list(X_umap_2d[:, 1].squeeze())
    df_2d['标注类别名称'] = g_label
    df_2d['图像索引'] = g_choics

    fig = px.scatter(
        df_2d, 
        x='X', 
        y='Y',
        color='标注类别名称',
        labels='标注类别名称',
        symbol='标注类别名称',
        hover_name='图像索引',
        opacity=0.8,
        width=width,
        height=height
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.write_html(f'{save_name}.html')


def save_to_picture(save_name, input_trans, groundtrues, text_size=48, markers=None, palette=None):
    if markers is None:
        markers = [
            '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 
            'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 
            5, 6, 7, 8, 9, 10, 11
        ]
    n_class = 30
    palette = sns.hls_palette(n_class, l=.4, s=.8)
    # palette = palette or sns.hls_palette(n_class) # 配色方案
    sns.palplot(palette)
    plt.figure(figsize=(12, 8))
    for idx, label in enumerate(np.unique(groundtrues)):
        # 获取颜色和点型
        color = palette[idx]
        marker = markers[idx%len(markers)]
        choices = input_trans[groundtrues==label]
        plt.scatter(choices[:, 0], choices[:, 1], color=color, marker=marker, label=label)
    plt.title('UMAP visualization of the data')
    plt.legend(loc="upper left")
    plt.savefig(f"/home/work/pytorch-toolkit-dev/dataset/data/{save_name}.png", dpi=300, bbox_inches="tight")
    print(save_name)

def save_to_picture1(save_name, input_trans, groundtrues, ex_input_trans, ex_groundtrues, text_size=48, markers=None, palette=None):
    if markers is None:
        markers = [
            '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 
            'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 
            5, 6, 7, 8, 9, 10, 11
        ]
    n_class = 30
    palette = sns.hls_palette(n_class, l=.4, s=.8)
    # palette = palette or sns.hls_palette(n_class) # 配色方案
    sns.palplot(palette)
    plt.figure(figsize=(12, 8))
    num = 7
    for idx, label in enumerate(np.unique(groundtrues)):
        # 获取颜色和点型
        color = palette[idx]
        marker = markers[idx%len(markers)]
        choices = input_trans[groundtrues==label]
        plt.scatter(choices[:, 0], choices[:, 1], color=color, marker=marker, label=label)
        num = idx + 1
    for idx, label in enumerate(np.unique(ex_groundtrues)):
        # 获取颜色和点型
        color = palette[idx+num]
        marker = markers[idx%len(markers)]
        choices = ex_input_trans[ex_groundtrues==label]
        plt.scatter(choices[:, 0], choices[:, 1], color=color, marker=marker, label=label)
    plt.title('UMAP visualization of the data')
    plt.legend(loc="upper left")
    plt.savefig(f"/home/work/pytorch-toolkit-dev/dataset/data/{save_name}.png", dpi=300, bbox_inches="tight")
    print(save_name)


def reduce_and_show(g_feats, g_label, g_files, choices, save_name, random_state=42, to_html=False):
    data_reducer= umap.UMAP(n_components=2, n_jobs=1, random_state=random_state)
    input_trans = data_reducer.fit_transform(g_feats)
    if to_html:
        save_to_html(save_name, input_trans, g_label, choices)
    else:
        save_to_picture(save_name, input_trans, g_label)
    
def reduce_and_show1(g_feats, g_label, g_files, ex_feats, ex_label, ex_files, choices, save_name, random_state=42, to_html=False):
    data_reducer= umap.UMAP(n_components=2, n_jobs=1, random_state=random_state)
    input_trans = data_reducer.fit_transform(g_feats)
    ex_input_trans = data_reducer.fit_transform(ex_feats)

    save_to_picture1(save_name, input_trans, g_label, ex_input_trans, ex_label)

def write_files(g_feats, g_label, g_files, class_name, name):
    if class_name == "PCA":
        # 降维到2D以便可视化（使用PCA）
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(g_feats)
    if class_name == "UMAP":
        data_reducer= umap.UMAP(n_components=2, n_jobs=1, random_state=42)
        data_2d = data_reducer.fit_transform(g_feats)

    # 获取唯一类别标签
    unique_labels = np.unique(g_label)

    # 计算每个类别的中心
    centers_2d = np.array([data_2d[g_label == unique_label].mean(axis=0) for unique_label in unique_labels])

    # 计算每个点到其类别中心的距离
    distances = np.array([np.linalg.norm(data_2d[i] - centers_2d[np.where(unique_labels == g_label[i])[0][0]]) for i in range(len(data_2d))])

    # 选择距离阈值
    distance_threshold = np.percentile(distances, 90)

    # 保存非离群点
    not_outlier_indices = np.where(distances < distance_threshold)[0]
    # 指定要写入的文件
    output_file = '/home/work/pytorch-toolkit-dev/dataset/data/train2/' + name + ".txt"
    # 将每个图片的路径和对应的标签按行写入到文档中
    with open(output_file, 'w', encoding='utf-8') as file:
        for i in tqdm(not_outlier_indices, desc="write files"):                   
            file.write(f'{g_files[i]},{g_label[i]}\n')

def draw_outlier(g_feats, g_label, g_files, class_name="UMAP", lable_name="PCA-val", liqun = "li"):
    if class_name == "PCA":
        # 降维到2D以便可视化（使用PCA）
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(g_feats)
    if class_name == "UMAP":
        data_reducer= umap.UMAP(n_components=2, n_jobs=1, random_state=42)
        data_2d = data_reducer.fit_transform(g_feats)
 
    # 获取唯一类别标签
    unique_labels = np.unique(g_label)

    # 计算每个类别的中心
    centers_2d = np.array([data_2d[g_label == unique_label].mean(axis=0) for unique_label in unique_labels])

    # 计算每个点到其类别中心的距离
    distances = np.array([np.linalg.norm(data_2d[i] - centers_2d[np.where(unique_labels == g_label[i])[0][0]]) for i in range(len(data_2d))])

    # 选择距离阈值
    distance_threshold = np.percentile(distances, 90)

    # 确定离群点  > 离群   < 非离群
    # outlier_indices = np.where(distances > distance_threshold)[0]
    
    if liqun == "li":
        outlier_indices = np.where(distances > distance_threshold)[0]
    if liqun == "not_li":
        outlier_indices = np.where(distances < distance_threshold)[0]

    # 打印远离类中心的图片路径   
    # for i in outlier_indices:
    #     target_folder = "/home/work/pytorch-toolkit-dev/dataset/data/vis/feats/"+ lable_name + "/" + g_label[i]
    #     if not os.path.exists(target_folder):
    #         os.makedirs(target_folder)
    #     shutil.copy(g_files[i], target_folder)
    print(class_name,lable_name,liqun)
    # 可视化
    plt.figure(figsize=(10, 7))
    num = 0
    # for unique_label in unique_labels:
    #     # 绘制类内的点
    #     plt.scatter(*data_2d[g_label == unique_label].T, label=f'{unique_label}')
    #     # 绘制远离中心的点
    #     outliers = (g_label == unique_label) & (distances > distance_threshold)
    #     print(unique_label,outliers.sum())
    #     num = num + outliers.sum()
    #     plt.scatter(*data_2d[outliers].T, color='k', marker='x')

    colors = plt.get_cmap('tab20', len(unique_labels))
    legend_handles = []  # 存储图例句柄

    for i, unique_label in enumerate(unique_labels):
        in_class = g_label == unique_label
        class_distances = distances[in_class]
        avg_distance = np.mean(class_distances)  # 计算平均距离
        
        # 绘制类内的点
        plt.scatter(*data_2d[in_class].T, color=colors(i))
        
        outliers = in_class & (distances > distance_threshold)
        outlier_distances = distances[outliers]
        avg_outlier_distance = np.mean(outlier_distances) if len(outlier_distances) > 0 else 0
        
        # # 仅当存在离群点时绘制
        # if len(outlier_distances) > 0:
        #     plt.scatter(*data_2d[outliers].T, color='k', marker='x')

        # 创建自定义图例项
        legend_text = f'{unique_label} (Avg Dist: {avg_distance:.2f}'
        if len(outlier_distances) > 0:
            legend_text += f', Outliers Avg Dist: {avg_outlier_distance:.2f})'
        else:
            legend_text += ')'
        legend_handles.append(Line2D([0], [0], marker='o', color='w', label=legend_text,
                                    markerfacecolor=colors(i), markersize=10))

    # 如果存在离群点，添加一个统一的离群点图例项
    # if any(distances > distance_threshold):
    #     legend_handles.append(Line2D([0], [0], marker='x', color='w', label='Outliers',
    #                                 markerfacecolor='k', markersize=10))

    # plt.legend(handles=legend_handles, loc='upper right', title='Class and Outliers')

    # print(g_feats.shape,num)
    # 绘制类中心
    plt.scatter(*centers_2d.T, color='r', marker='*', s=200, label='中心')
    # plt.legend()
    plt.title('特征可视化与离群点标记')
    # plt.savefig(f"/home/work/pytorch-toolkit-dev/dataset/data/vis/feats/"+ lable_name + ".png" , dpi=300, bbox_inches="tight")
    plt.savefig(f"test_jiqing.png" , dpi=300, bbox_inches="tight")
    # # plt.show()
    # # find /home/work/pytorch-toolkit-dev/dataset/data/outlier/PCA-val -type f | wc -l

    

if __name__=="__main__":
    gallerys = "dataset/data/vis/vis2/18o_on_shai1-val.npz"  # 18o_on_shai1-val
    # gallerys = "/home/work/pytorch-toolkit-dev/dataset/data/vis/18_on_new-val.npz"
    # gallerys = "/home/work/pytorch-toolkit-dev/dataset/data/blacklist-val.npz"
    label_file = "dataset/zero_dataset/label_names.csv"
    g_feats, g_label, g_files = load_data(gallerys)
    
    # write_files(g_feats, g_label, g_files, "PCA", "train")

    n_class = 30
    choose_cats = list(set(g_label))[:n_class]
    choose_cats = [999921949, 9999150308, 999925336, 10253, 999920421, 999920839, 999921427]
    # choose_cats = [999921148, 999925224, 999921064, 9999150390, 999920460, 999920889]
    # choose_cats = [999925336, 10253, 999920421]
    

    choose_cats = [999920139,9999150860,999920220,9999150700,999920229,999920814,9999150675,9999150257,999920865,9999150553]
    choose_cats = [9999151980,999920310,999920751,999920226,999920421,999921337,9999150675,999921028,999921403,9999151860,999925357,9999150365,999925336,9999150642]
    choices = np.where(np.isin(g_label, choose_cats) == True)[0]
    g_feats, g_label, g_files = g_feats[choices], g_label[choices], g_files[choices]
    faiss.normalize_L2(g_feats)

    label_index = load_csv_file(label_file)
    label_map = {int(cat): name.split("/")[0] for cat, name in label_index.items()}
    g_label = np.array([label_map[l] for l in g_label])
    # reduce_and_show(g_feats, g_label, g_files, choices, "test_umap6", to_html=False)
    
    # draw_outlier(g_feats, g_label, g_files, "UMAP", "UMAP-val2","li")
    # draw_outlier(g_feats, g_label, g_files, "UMAP", "UMAP-val1","not_li")
    draw_outlier(g_feats, g_label, g_files, "PCA", "18o_on_shai1_evl","li")
    # draw_outlier(g_feats, g_label, g_files, "PCA", "PCA-val1","not_li")

    # g_feats[np.where(np.isin(g_label, [999921427]) == True)[0]].shape  
    # "999921949","串西红柿/串红/红串柿/串红西红柿/枝纯千禧/串收樱桃番茄（红）198g/千禧番茄/樱桃小番茄/枝纯红番茄/串番茄"    467     114 
    # "9999150308","前尖猪肉/精前腿肉/前腿肉/前臀尖/前肩肉/前夹肉/精品前尖/前尖"     430                                  31
    # "999925336","散称桂皮/肉桂/桂皮z/桂皮散/香桂/桂皮（散）/桂皮/散桂皮/桂皮（散）/干桂皮/桂心"    309  
    # "10253","草莓柿子/水果西红柿/草莓番茄/樱桃柿子/草莓西红柿/中西红柿/绿水峡谷水果西红柿/粉红西红柿"    488
    # "999920421","柿子/火晶柿子/火柿子/软柿子/莲花柿"                                  396     
    # "999920839","贝贝小柿子/贝贝柿子/玲珑小番茄/玲珑小柿子/                               488
    # "999921427","柿子椒/青椒/湖南青椒/菜椒/青圆椒/青元椒/圆青椒/柿椒/大青椒/精品青椒/青大椒/青圆泡椒/青灯笼椒/青椒（农）/山东灯椒/圆椒"   453 

    # find ./dataset/optimize_task3/images -type d -name "999921427"
    # find ./dataset/data/outlier/PCA-val1/丝瓜 -type f | wc -l
    # 
 







    
