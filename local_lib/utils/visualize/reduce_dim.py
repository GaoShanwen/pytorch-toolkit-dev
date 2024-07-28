import umap
import faiss
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from local_lib.utils.file_tools import load_data, load_csv_file
import seaborn as sns

from mplfonts import use_font
use_font('Noto Sans CJK SC')#指定中文字体


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
    palette = palette or sns.hls_palette(n_class) # 配色方案
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
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches="tight")


def reduce_and_show(g_feats, g_label, g_files, choices, save_name, random_state=42, to_html=False):
    data_reducer= umap.UMAP(n_components=2, n_jobs=1, random_state=random_state)
    input_trans = data_reducer.fit_transform(g_feats)
    if to_html:
        save_to_html(save_name, input_trans, g_label, choices)
    else:
        save_to_picture(save_name, input_trans, g_label)
    

if __name__=="__main__":
    gallerys = "output/feats/regnety_040-train-inter_similarity-0.945.npz"
    label_file = "dataset/zero_dataset/label_names.csv"
    g_feats, g_label, g_files = load_data(gallerys)
    n_class = 3
    choose_cats = list(set(g_label))[:n_class]
    choices = np.where(np.isin(g_label, choose_cats) == True)[0]
    g_feats, g_label, g_files = g_feats[choices], g_label[choices], g_files[choices]
    faiss.normalize_L2(g_feats)

    label_index = load_csv_file(label_file)
    label_map = {int(cat): name.split("/")[0] for cat, name in label_index.items()}
    g_label = np.array([label_map[l] for l in g_label])
    reduce_and_show(g_feats, g_label, g_files, choices, "test_umap", to_html=False)
