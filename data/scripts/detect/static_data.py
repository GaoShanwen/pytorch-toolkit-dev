import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取表格数据，第一行作为列名
df = pd.read_csv('data/det-dataset/hld/data_info.csv')
df['image_count'] = pd.to_numeric(df['image_count'])

# 获取任务名称的唯一值
tasks = ["train","val"]#df['task'].unique()

# 定义属性列
attributes = ['angle','is locate black','is locate bright','is wall disorder','is road disorder','is crowed','is single road']

# 统计每个任务下各属性的文件夹数量和图片数量
results = {}
for task in tasks:
    task_data = df[df['task'] == task]
    task_results = {}
    for attr in attributes:
        # 统计文件夹数量（元素出现次数）
        folder_count = task_data[attr].value_counts().sort_index()
        # 统计图片数量（image_count 的总和）
        image_count = task_data.groupby(attr)['image_count'].sum().sort_index()
        task_results[attr] = {
            'folder_count': folder_count,
            'image_count': image_count
        }
    results[task] = task_results

# 绘制柱状图
def plot_bar_chart(data, title, ylabel, filename):
    fig, axes = plt.subplots(nrows=1, ncols=len(attributes), figsize=(15, 5), sharey=True)
    history_ymax = 0
    for i, attr in enumerate(attributes):
        unique_values = sorted(set(df[attr].dropna().unique()))
        unique_values = unique_values if len(unique_values)>1 else [0, 1]
        x = np.arange(len(unique_values))

        task1_values = data[tasks[0]][attr].reindex(unique_values, fill_value=0)
        task2_values = data[tasks[1]][attr].reindex(unique_values, fill_value=0)
        
        
        # 绘制柱状图
        axes[i].bar(x - 0.2, task1_values, width=0.8, label=tasks[0], color='blue')
        axes[i].bar(x - 0.2, task2_values, bottom=task1_values, width=0.8, label=tasks[1], color='orange')
        # axes[i].bar(x - 0.2, task2_values, width=0.8, label=f'{tasks[1]}', color='blue')
        # axes[i].bar(x - 0.2, task1_values, bottom=task2_values, width=0.8, label=f'{tasks[0]}', color='orange')
        
        # 设置X轴标签和标题
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(unique_values, rotation=45, ha='right')
        axes[i].set_title(f'{attr}')
        if not i:
            axes[i].set_ylabel(ylabel)
        y_max = (task1_values+task2_values).max()  # 获取数据的最大值
        if y_max > history_ymax:
            history_ymax = y_max
        # axes[i].legend()
    axes[0].set_ylim(0, history_ymax * 1.1)  # 设置 y 轴范围，最大值增加 10% 的余量
    # 调整布局
    plt.tight_layout()
    
    # 保存图表为图片文件
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存为 {filename}")

# 绘制文件夹数量的柱状图
folder_data = {task: {attr: results[task][attr]['folder_count'] for attr in attributes} for task in tasks}
plot_bar_chart(folder_data, 'Folder Count', 'Number of Folders', 'folder_count_plot.png')

# 绘制图片数量的柱状图
image_data = {task: {attr: results[task][attr]['image_count'] for attr in attributes} for task in tasks}
plot_bar_chart(image_data, 'Image Count', 'Number of Images', 'image_count_plot.png')
