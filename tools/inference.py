'''
2024.11.08
'''

import os
os.environ["YOLO_VERBOSE"] = "false"
import argparse
import datetime
import time
from collections import deque

import cv2
import torch
import numpy as np
from ultralytics import YOLO

from stream import start


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='video url')
    parser.add_argument('--weights', type=str, required=True, help='weights path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--interval', type=int, default=1, help='sample interval')
    parser.add_argument('--json_port', type=int, required=True, help='json port')
    parser.add_argument('--rtsp_port', type=int, help='rtsp port')
    parser.add_argument('--mjpeg_port', type=int, help='mjpeg port')
    parser.add_argument('--factor', type=float, default=0.7, help='compression factor')
    return parser.parse_args()


def detect(add_line, lines, color, width):
    model = YOLO(opt.weights)
    names = model.names

    img = torch.zeros((1, 3, opt.img_size, opt.img_size))
    model(img)

    data_deque = deque(maxlen=100)
    start(data_deque, opt.json_port, opt.rtsp_port, opt.mjpeg_port, out_size=(out_width, out_height), max_length=100)

    cap = cv2.VideoCapture(opt.video)
    frame_id = 0
    reconnected = False
    objects = []

    start_time = time.time()
    while True:
        t0 = time.time()
        _, frame = cap.read()
        if frame is None:
            if reconnected:
                break
            cap.open(opt.video)
            reconnected = True
            continue

        frame_id += 1

        if frame_id % opt.interval == 0:
            # 清空和初始化操作
            objects = []
            # boxes = None
            results = model(frame, conf=opt.conf_thres, imgsz=opt.img_size, half=opt.half)[0]
            if len(results) > 0:
                # cv2.imwrite('origin.jpg', frame)
                # frame = results.plot() 
                # 获取box信息
                boxes = results.boxes
                cls = boxes.cls.int().tolist()
                conf = boxes.conf.tolist()
                xywhn = boxes.xywhn.tolist()
                # 获取掩码信息
                masks = results.masks
                masks_xy = masks.xy

                # 打包发送的信息  ============================ 
                for i in range(len(boxes)):
                    obj = {"class_id": cls[i], 
                            "name": names[cls[i]], 
                            "confidence": conf[i],
                            "relative_coordinates":{
                                "center_x": xywhn[i][0],
                                "center_y": xywhn[i][1], 
                                "width": xywhn[i][2],
                                "height": xywhn[i][3]},
                            "masks_xy": masks_xy[i].tolist(),
                    }
                    objects.append(obj)
                frame = results.plot()
                # 打包信息结束 ============================== 


        json = {"frame_id": frame_id,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "objects": objects
        }
        # 绘制候选区域
        if add_line is not None:
            for line, c, w in zip(lines, color, width):
                cv2.polylines(frame, [np.array(line)], isClosed=True, color=c, thickness=w)
        data_deque.append((json, frame))

        print('fps: %.3f, \tavg_fps: %.3f, \tbuffer: %g' % (1 / (time.time() - t0), frame_id / (time.time() - start_time), len(data_deque)))


if __name__ == '__main__':
    opt = parse_args()
    print(opt)

    # 从配置文件中获取out_size
    import json
    cwd = os.getcwd()[7:]
    business_json_list = ["business.json", "/root/MBAB/AI/{}/etc/business.json".format(cwd)]
    out_width = 1280
    out_height = 720
    add_line = lines = color = width = None
    for business_json in business_json_list:
        if os.path.isfile(business_json):
            business = json.load(open(business_json))
            params = business["business_params"]
            out_size = params.get("out_size")
            if out_size is not None:
                out_width = out_size.get("out_width")
                out_height = out_size.get("out_height")
                
            add_line = params.get("add_line", None)
            if add_line is not None:
                lines = add_line.get("lines")
                width = add_line.get("width")
                color = add_line.get("color")
            break
    # 读取配置文件 end
    detect(add_line, lines, color, width)


