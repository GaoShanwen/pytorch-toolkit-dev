# coding=utf-8
import argparse
import os
import cv2
import time
from tqdm import tqdm
import numpy as np
from multiprocessing import Manager, Process
 

def parse_args():
    parser = argparse.ArgumentParser(description='Download data from urls file')
    parser.add_argument('-v', '--video-dir', type=str, required=True, help='the dir of ori videos')
    parser.add_argument('-i', '--image-root', type=str, required=True, help='the root of dst images')
    parser.add_argument('-s', '--img-size', type=int, default=None, nargs='*', help='the width and height of img')
    parser.add_argument('-f', '--formats', type=str, nargs='*', default=[".mp4", ".MOV"], help='supportted formats')
    parser.add_argument('-g', '--time-interval', type=int, default=10, help='the gap of sample from video')
    return parser.parse_args()


def saved_files(video_path, obj_dir, video_name, interval, size, progress, saved, failed):
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    if cap.isOpened():
        success = True
    else:
        success = False
        print(f"视频{video_name}, 读取失败!")

    while(success):
        success, frame = cap.read()
        if frame_index % interval == 0:
            try:
                if size:
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                img_name=f"{video_name}-{frame_index:06d}.jpg"
                cv2.imwrite(os.path.join(obj_dir, img_name), frame)
                saved.value += 1
            except Exception as e:
                failed.value += 1
                pass
        frame_index += 1
    cap.release()
    progress.value += 1


def update_progress(progress, saved, failed, num_total):
    last_value = 0
    prefix = "获取图片/获取失败量: "
    with tqdm(total=num_total, desc=f"{prefix}{saved.value}/{failed.value}", unit="element") as pbar:
        while progress.value < num_total:
            pbar.n = progress.value  # 设置当前进度
            pbar.last_print_n = progress.value  # 强制更新进度条
            pbar.set_description(f"{prefix}{saved.value}/{failed.value}")
            pbar.update(progress.value-last_value)
            last_value = progress.value
            time.sleep(0.1)
        pbar.set_description(f"{prefix}{saved.value}/{failed.value}")
        pbar.update(progress.value-last_value)


def video2frame(video_src_path, frame_save_path, frame_size, interval, formats, workers=48):
    videos = [v for v in os.listdir(video_src_path) if v[-4:] in formats]
    frame_index = 0
    total = len(videos)
    with Manager() as manager:
        progress = manager.Value('i', 0)
        saved = manager.Value('i', 0)
        failed = manager.Value('i', 0)

        progress_thread = Process(target=update_progress, args=(progress, saved, failed, total))
        progress_thread.start()

        processes = []
        for each_video in videos:
            video_name = each_video[:-4]
            obj_dir = os.path.join(frame_save_path, video_name)
            if not os.path.exists(obj_dir):
                os.mkdir(obj_dir)
    
            path = os.path.join(video_src_path, each_video)
            values = (path, obj_dir, video_name, interval, frame_size, progress, saved, failed)
            p = Process(target=saved_files, args=values)
            p.start()
            processes.append(p)

            if len(processes) >= workers:
                processes[0].join()
                processes = processes[1:]
        
        for p in processes:
            p.join()
        progress_thread.join()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.image_root):
        os.makedirs(args.image_root, exist_ok=True)
    video2frame(args.video_dir, args.image_root, args.img_size, args.time_interval, args.formats)
