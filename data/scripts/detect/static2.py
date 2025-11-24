import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import yaml


def parse_yolo_annotations(annot_dir, img_size=(640, 640)):
    """
    解析YOLO格式标注文件，提取所有实例的归一化宽高
    
    Args:
        annot_dir: 标注文件目录（.txt文件）
        img_size: 图像默认尺寸（w, h），用于验证归一化范围（可选）
    
    Returns:
        widths: 所有实例的归一化宽度列表
        heights: 所有实例的归一化高度列表
    """
    widths = []
    heights = []
    
    # 遍历所有标注文件
    with open(annot_dir, 'r', encoding='utf-8') as f:
        annot_list = [line.strip().replace(".jpg", ".txt") for line in f.readlines()]
    static_dict = {d:{"w":[], "h":[]} for d in range(13)}
    for txt_file in annot_list:
        if not txt_file.endswith('.txt') or not os.path.exists(txt_file):
            continue
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # YOLO格式：class_id x_center y_center w h
            parts = line.split()
            if len(parts) != 6:
                print(f"警告：{txt_file} 中存在格式错误的行：{line}")
                continue
            
            # 提取归一化宽高（已归一化到0-1范围）
            cat = int(parts[0])
            w = float(parts[3])
            h = float(parts[4])
            
            # 过滤异常值（可选）
            if 0 < w <= 1 and 0 < h <= 1:
                # widths.append(w)
                # heights.append(h)
                static_dict[cat]["w"].append(w)
                static_dict[cat]["h"].append(h)
            else:
                print(f"警告：{txt_file} 中存在异常宽高值（w={w}, h={h}），已过滤")
    
    # return np.array(widths), np.array(heights)
    return static_dict

def hex_to_rgb(hex_color):
    """
    将6位16进制颜色字符串转换为RGB值（0→1范围）
    
    Args:
        hex_color: 6位16进制颜色（如 '#FFFFFF'）
    
    Returns:
        rgb: (r, g, b) 元组（每个值0-1）
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def create_white_zero_to_blue_cmap():
    """
    用6位16进制颜色创建：仅密度=0为白色，密度>0从浅蓝→深蓝渐变的colormap
    配色逻辑：白色（仅密度0）→ 浅蓝→天蓝→深蓝→藏青（密度>0递增）
    """
    # 6位16进制颜色定义（关键：第一个节点为白色，后续全为蓝色系）
    hex_colors = [
        '#FFFFFF',  # 仅密度=0（无实例）：纯白色
        '#B3E0FF',  # 密度>0起始：天蓝（低密度实例）
        '#66B2FF',  # 蓝色（中密度）
        '#1A8CFF',  # 深蓝（中高密度）
        '#0047AB'   # 最大值：藏青蓝（最高密度）
    ]
    
    # 颜色节点位置（重点：白色仅占0.0一个点，后续蓝色系均匀分布在0.0→1.0）
    # 确保密度>0时立即进入蓝色系，无白色过渡
    positions = [0.0, 0.001, 0.4, 0.7, 1.0]
    
    # 转换为RGB值（0→1范围）
    rgb_colors = [hex_to_rgb(hex_color) for hex_color in hex_colors]
    
    # 构建LinearSegmentedColormap所需的字典格式
    cmap_dict = {
        'red': [(pos, rgb[0], rgb[0]) for pos, rgb in zip(positions, rgb_colors)],
        'green': [(pos, rgb[1], rgb[1]) for pos, rgb in zip(positions, rgb_colors)],
        'blue': [(pos, rgb[2], rgb[2]) for pos, rgb in zip(positions, rgb_colors)]
    }
    
    # 创建256级渐变的colormap（确保蓝色系过渡平滑）
    return LinearSegmentedColormap('white_zero_to_blue', cmap_dict, N=256)

def create_yolo_heatmap(widths, heights, save_path='yolo_wh_heatmap.png', 
                        bins=50, figsize=(10, 8), xlim=(0, 1), ylim=(0, 1)):
    """
    绘制YOLO工程风格热力图：仅无实例为白色，有实例从浅蓝→深蓝渐变
    
    Args:
        widths: 归一化宽度数组
        heights: 归一化高度数组
        save_path: 图片保存路径
        bins: 网格数量（越大越精细）
        figsize: 图片尺寸
        xlim/ylim: 坐标轴范围
    """
    # 设置中文字体（避免中文乱码）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 计算2D直方图（核心：统计每个宽高网格的实例数量）
    hist, xedges, yedges = np.histogram2d(
        widths, heights, bins=bins, range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]]
    )
    
    # 转置直方图：确保x轴为宽度、y轴为高度（匹配YOLO坐标习惯）
    hist = hist.T
    
    # 创建自定义配色（仅密度0为白色，密度>0→蓝色渐变）
    cmap = create_white_zero_to_blue_cmap()
    
    # 创建画布和子图（高dpi确保清晰度）
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    
    # 绘制热力图（关键：vmin=0锁定白色为密度0，vmax自动适配最大密度）
    im = ax.imshow(
        hist,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='lower',  # 左下角为(0,0)，符合YOLO归一化坐标
        cmap=cmap,
        aspect='auto',
        vmin=0  # 强制密度0对应白色
    )
    
    # 设置坐标轴标签和标题（YOLO风格简洁明了）
    ax.set_xlabel('归一化宽度 (Normalized Width)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('归一化高度 (Normalized Height)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(
        f'YOLO数据集宽高分布热力图\n(总实例数: {len(widths)}, 网格数: {bins}x{bins})',
        fontsize=14, fontweight='bold', pad=20
    )
    
    # 添加网格线（浅灰色，增强可读性且不干扰颜色）
    ax.grid(True, alpha=0.3, linestyle='--', color='#999999')
    
    # 添加颜色条（标注密度值，说明白色为无实例）
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('实例密度 (Instance Density) | 白色=无实例', fontsize=10, fontweight='bold')
    
    # 自定义坐标轴刻度（0→1均匀分布6个刻度）
    ax.set_xticks(np.linspace(xlim[0], xlim[1], 6))
    ax.set_xticklabels([f'{x:.1f}' for x in np.linspace(xlim[0], xlim[1], 6)], fontsize=10)
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 6))
    ax.set_yticklabels([f'{y:.1f}' for y in np.linspace(ylim[0], ylim[1], 6)], fontsize=10)
    
    # 增强边框效果（让图表更规整）
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    
    # 调整布局（避免标签被截断）
    plt.tight_layout()
    
    # 保存高清图片（白色背景，适配各类展示场景）
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"热力图已保存至：{save_path}")
    # 显示图片（可选，运行时预览）
    plt.show()


def create_multi_class_heatmap(class_wh, num_classes=13, save_path='yolo_14class_wh_heatmap.png', 
                               bins=30, figsize=(28, 8), xlim=(0, 1), ylim=(0, 1), cats=[]):
    """
    绘制14类别YOLO宽高分布热力图（2行7列布局）
    
    Args:
        class_wh: 字典，key=类别ID，value=(widths, heights)
        num_classes: 类别总数（固定14）
        save_path: 图片保存路径
        bins: 每个子图的网格数量（默认30x30，适配多子图布局）
        figsize: 整体图片尺寸（28x8适配2x7布局）
        xlim/ylim: 坐标轴范围
    """
    # 设置中文字体（避免中文乱码）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建自定义配色
    cmap = create_white_zero_to_blue_cmap()
    
    # 创建2行7列的子图布局
    fig, axes = plt.subplots(nrows=2, ncols=7, figsize=figsize, dpi=150, squeeze=False)
    axes = axes.flatten()  # 转为1维数组，便于循环操作
    
    # 计算所有类别中的最大密度（用于统一颜色条范围，确保所有子图配色一致）
    max_density = [0]*num_classes
    class_hists = {}
    for cid in range(num_classes):
        widths, heights = class_wh[cid]["w"], class_wh[cid]["h"]
        if len(widths) == 0:
            class_hists[cid] = np.zeros((bins, bins))
            continue
        # 计算当前类别的2D直方图
        hist, _, _ = np.histogram2d(widths, heights, bins=bins, range=[xlim, ylim])
        hist = hist.T  # 转置适配坐标轴
        class_hists[cid] = hist
        # 更新最大密度
        current_max = hist.max()
        if current_max > max_density[cid]:
            max_density[cid] = current_max
    
    # 绘制每个类别的热力图
    for cid in range(num_classes):
        ax = axes[cid]
        widths, heights = class_wh[cid]["w"], class_wh[cid]["h"]
        hist = class_hists[cid]
        num_instances = len(widths)
        
        # 绘制热力图（vmin=0，vmax=max_density统一配色范围）
        im = ax.imshow(
            hist,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            origin='lower',
            cmap=cmap,
            aspect='auto',
            vmin=0,
            vmax=max_density[cid]  # 所有子图使用相同的最大值，确保颜色可比
        )
        # 设置子图标题（类别ID + 实例数）
        ax.set_title(f'CAT: {cats[cid]} (NUM INS: {num_instances}) ', 
                     fontsize=11, fontweight='bold', pad=10)
        
        # 设置坐标轴标签（仅第一列和最后一行显示，避免重复）
        if cid % 7 == 0:  # 第一列显示y轴标签
            ax.set_ylabel('Normal Height (H)', fontsize=10, fontweight='bold')
        if cid >= 7:  # 第二行显示x轴标签
            ax.set_xlabel('Normal Width (W)', fontsize=10, fontweight='bold')
        
        # 自定义坐标轴刻度（0→1均匀分布4个刻度，避免多子图拥挤）
        tick_pos = np.linspace(xlim[0], xlim[1], 4)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([f'{x:.1f}' for x in tick_pos], fontsize=8)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels([f'{y:.1f}' for y in tick_pos], fontsize=8)
        
        # 添加网格线（浅灰色，增强可读性）
        ax.grid(True, alpha=0.3, linestyle='--', color='#999999')
        
        # 增强边框
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('#333333')
    
    # # 添加全局颜色条（所有子图共享，放在右侧）
    # cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, orientation='vertical')
    # cbar.set_label('Instance Density ', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
    
    # 设置全局标题
    fig.suptitle(f'Vehicle Train (NUM SIN: {bins}x{bins}) ',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 调整子图间距（避免重叠）
    plt.tight_layout(rect=[0, 0, 0.97, 0.95])  # 预留颜色条空间
    
    # 保存高清图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"14类别热力图已保存至: {save_path}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLO数据集宽高分布热力图（仅无实例为白色→蓝色渐变）')
    parser.add_argument('--annot-dir', required=True, help='YOLO标注文件目录（.txt文件）')
    parser.add_argument('--save-path', default='yolo_wh_heatmap_{}.png', help='热力图保存路径')
    parser.add_argument('--task', help='任务名称')
    parser.add_argument('--bins', type=int, default=50, help='热力图网格数量（默认50x50，越大越精细）')
    parser.add_argument('--img-size', type=int, nargs=2, default=(640, 640), help='图像尺寸（w, h），仅用于验证标注')
    args = parser.parse_args()
    
    with open(os.path.join("data/det-dataset", args.task, "dataset.yaml"), 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        cats = [v for _, v in data["names"].items()]
    # 1. 解析标注文件，提取宽高数据
    print(f"正在解析标注文件：{args.annot_dir}")
    static_dict = parse_yolo_annotations(args.annot_dir, args.img_size)
    create_multi_class_heatmap(class_wh=static_dict, save_path="yolo_wh_heatmap.png", bins=args.bins, cats=cats)

if __name__ == '__main__':
    main()