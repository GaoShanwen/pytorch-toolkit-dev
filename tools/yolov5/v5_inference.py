import argparse
import os
import torch
import json
from collections import deque
import time

from ultralytics.utils.plotting import Annotator, colors

from models.common import DetectMultiBackend
from utils.general import (
    cv2,
    non_max_suppression,
    print_args,
    scale_boxes,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
from stream import start

import numpy as np
import datetime
import yaml


@smart_inference_mode()
def run(
    weights,  # model path or triton URL
    source,  # file/dir/URL/glob/screen/0(webcam)
    data,
    json_port,
    rtsp_port,
    mjpeg_port,
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    interval=1,  # video frame-rate stride
):

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fp16=half)

    with open(data, errors='ignore') as f:
        names = yaml.safe_load(f)['names']

    torch.backends.cudnn.benchmark = True

    # Run inference
    model.warmup(imgsz=(1 , 3, *imgsz))  # warmup
    data_deque = deque(maxlen=50)

    torch.cuda.empty_cache()

    start(data_deque, json_port, rtsp_port, mjpeg_port, out_size=None, fps=25, max_length=50, timeout=10)

    frame_id = 0
    objects = []

    param = cv2.cudacodec.VideoReaderInitParams()
    param.udpSource = True
    video_reader = cv2.cudacodec.createVideoReader(source, params=param)
    stream = cv2.cuda_Stream()
    orig_size1 = None
    orig_size2 = None
    global lines1
    global lines2
    # reconnected = False
    
    print(out_width, out_height)

    start_time = time.time()
    while True:
        t0 = time.time()
        ret, frame_gpu = video_reader.nextFrame()
        t1 = time.time()
        if not ret or frame_gpu is None:
            # if reconnected:
            #     break
            param = cv2.cudacodec.VideoReaderInitParams()
            param.udpSource = True
            video_reader = cv2.cudacodec.createVideoReader(source, params=param)
            stream = cv2.cuda_Stream()
            # reconnected = True
            print("cuda video_reader reconnect...")
            continue
        
        frame_bgr_gpu = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGRA2BGR, stream=stream)
        if not orig_size1 and add_line1:
            orig_size1 = frame_bgr_gpu.size()
            lines1 = (np.asarray(lines1) / orig_size1 * (out_width, out_height)).astype(int)
        
        if not orig_size2 and add_line2:
            orig_size2 = frame_bgr_gpu.size()
            lines2 = (np.asarray(lines2) / orig_size2 * (out_width, out_height)).astype(int)
            
        frame_bgr_gpu_outsize = cv2.cuda.resize(frame_bgr_gpu, (out_width, out_height), stream=stream)
        frame = frame_bgr_gpu_outsize.download(stream=stream)

        t2 = time.time()
        frame_id += 1

        if frame_id % interval == 0:
            shape = frame.shape[:2]  # current shape [height, width]

            # Scale ratio (new / old)
            r = min(imgsz[0] / shape[0], imgsz[1] / shape[1])

            # Compute padding
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = imgsz[1] - new_unpad[0], imgsz[0] - new_unpad[1]  # wh padding

            dw /= 2  # divide padding into 2 sides
            dh /= 2
            im = cv2.cuda.resize(frame_bgr_gpu, new_unpad, stream=stream)
            im = im.download(stream=stream)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border

            im = np.ascontiguousarray(im[np.newaxis, ..., ::-1].transpose((0, 3, 1, 2)))  # BGR to RGB, BHWC to BCHW, contiguous

            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0

            t3 = time.time()
            pred = model(im)
            t4 = time.time()

            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process predictions
            det = pred[0]
            gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            objects = []

            if len(det):
                # Rescale boxes from img_size to frame size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    obj = {"class_id": c, 
                        "name": names[c], 
                        "confidence": float(conf),
                        "relative_coordinates":{
                            "center_x": xywh[0],
                            "center_y": xywh[1], 
                            "width": xywh[2],
                            "height": xywh[3]}
                    }

                    objects.append(obj)

                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    annotator.box_label(xyxy, label, color=colors(c, True))

            t5 = time.time()
            print('read: %.3f, download: %.3f, \tprep: %.3f, \tinfer: %.3f, \tpost: %.3f' % (1/(t1-t0), 1/(t2-t1), 1/(t3-t2), 1/(t4-t3), 1/(t5-t4)))
        print('fps: %.3f, \tavg_fps: %.3f, \tbuffer: %g' % (1 / (time.time() - t0), frame_id / (time.time() - start_time), len(data_deque)))

        if add_line1:
            for line, c, w in zip(lines1, color1, width1):
                cv2.polylines(frame, [np.array(line)], isClosed=True, color=c, thickness=w)
                if filling1:
                    mask = np.zeros(frame.shape, np.uint8)
                    mask = cv2.fillPoly(mask, np.array([line]), tuple(c))
                    frame = cv2.addWeighted(src1=frame, alpha=1 - alpha1, src2=mask, beta=alpha1, gamma=0)
        if add_line2:
            for line, c, w in zip(lines2, color2, width2):
                cv2.polylines(frame, [np.array(line)], isClosed=True, color=c, thickness=w)
                if filling2:
                    mask = np.zeros(frame.shape, np.uint8)
                    mask = cv2.fillPoly(mask, np.array([line]), tuple(c))
                    frame = cv2.addWeighted(src1=frame, alpha=1 - alpha2, src2=mask, beta=alpha2, gamma=0)
                
        json = {"frame_id": frame_id,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "objects": objects
        }
        data_deque.append((json, frame))



def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov5m.pt", help="model path")
    parser.add_argument("--source", type=str, help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument('--data', type=str, default='yolov5.yaml', help='dataset.yaml path')
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=50, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--interval", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--json_port", type=int,  help="json port")
    parser.add_argument("--rtsp_port", type=int,  help="rtsp port")
    parser.add_argument("--mjpeg_port", type=int,  help="mjpeg port")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


if __name__ == "__main__":
    opt = parse_opt()

    cwd = os.getcwd()[7:]
    business_json_list = ["business.json", "/root/MBAB/AI/{}/etc/business.json".format(cwd)]
    out_width = 1280
    out_height = 720
    add_line1 = lines1 = color1 = width1 = None
    add_line2 = lines2 = color2 = width2 = None

    for business_json in business_json_list:
        if os.path.isfile(business_json):
            business = json.load(open(business_json))
            params = business["business_params"]
            out_size = params.get("out_size")
            if out_size is not None:
                out_width = out_size.get("out_width")
                out_height = out_size.get("out_height")
                
            add_line1 = params.get("add_line1")
            if add_line1 is not None:
                lines1 = add_line1.get("lines1")
                line1 = add_line1.get("line1")
                lines1 = lines1 or [line1]
                width1 = add_line1.get("width1")
                color1 = add_line1.get("color1")
                alpha1 = add_line1.get("alpha1")
                filling1 = add_line1.get("filling1")

            add_line2 = params.get("add_line2")
            if add_line2 is not None:
                lines2 = add_line2.get("lines2")
                width2 = add_line2.get("width2")
                color2 = add_line2.get("color2")
                alpha2 = add_line2.get("alpha2")
                filling2 = add_line2.get("filling2")
                
    run(**vars(opt))
