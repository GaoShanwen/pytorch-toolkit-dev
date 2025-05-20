import cv2
import os
import numpy as np
from tqdm import tqdm


def images_to_video(image_folder, output_video, fps=30, frame_size=None):
    images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    images.sort()  # 按文件名排序

    if not images:
        raise ValueError("图片文件夹中没有图片文件！")

    # 读取第一张图片以获取帧大小
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    if frame_size is None:
        frame_size = (first_image.shape[1], first_image.shape[0])  # (宽度, 高度)

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    # 遍历所有图片并写入视频
    for image_name in tqdm(images):
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)

        # 调整图片大小以匹配视频帧大小
        if frame_size:
            frame = cv2.resize(frame, frame_size)

        out.write(frame)  # 写入帧

    # 释放视频写入对象
    out.release()
    print(f"视频已保存到: {output_video}")


if __name__ == '__main__':
    image_folder = "data/det-dataset/test/result/dv1"  # 图片文件夹路径
    output_video = "runs/detect/result1/dv1-1.mp4"  # 输出视频文件路径
    fps = 30  # 帧率
    images_to_video(image_folder, output_video, fps)