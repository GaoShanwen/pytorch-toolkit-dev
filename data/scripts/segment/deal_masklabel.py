import json
import os
import numpy as np
import cv2
from shapely.geometry import Polygon


def bbox_iou(box1, box2):
    x_maxA, x_maxB = max(box1[:, 0]), max(box2[:, 0])
    y_maxA, y_maxB = max(box1[:, 1]), max(box2[:, 1])
    x_minA, x_minB = min(box1[:, 0]), min(box2[:, 0])
    y_minA, y_minB = min(box1[:, 1]), min(box2[:, 1])
    inter = max(0, min(x_maxA, x_maxB) - max(x_minA, x_minB)) \
            * max(0, min(y_maxA, y_maxB) - max(y_minA, y_minB))
    area1 = (x_maxA - x_minA) * (y_maxA - y_minA)
    area2 = (x_maxB - x_minB) * (y_maxB - y_minB)
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0


def merge_polygons(polygons_list):
    if len(polygons_list) < 2:
        return np.vstack(polygons_list).reshape(-1, 2)
    merged_vertices = []
    start_p = None
    for poly in polygons_list:
        poly = np.vstack((poly, [poly[0]]))
        if start_p is None:
            start_p = poly[0]
        else:
            poly = np.vstack((poly, [start_p]))
        merged_vertices.extend(poly)  # 逐个添加顶点
    
    return np.array(merged_vertices).reshape(-1, 2)

def get_polygon(vertices, image_shape, d=0, cargo_poly=None):
    vertices_np = np.array(vertices, dtype=np.int32)
    x_min, y_min = vertices_np.min(axis=0)
    x_max, y_max = vertices_np.max(axis=0)
    
    if cargo_poly:
        cargos_np = np.array(cargo_poly, dtype=np.int32)
        cargo_x_min, cargo_y_min = cargos_np.min(axis=0)
        cargo_x_max, cargo_y_max = cargos_np.max(axis=0)
        x_min, y_min = min(x_min, cargo_x_min), min(y_min, cargo_y_min)
        x_max, y_max = max(x_max, cargo_x_max), max(y_max, cargo_y_max)

    x_start = max(0, x_min - d)
    y_start = max(0, y_min - d)
    x_end = min(image_shape[1], x_max + d + 1)
    y_end = min(image_shape[0], y_max + d + 1)
    roi_shape = (y_end - y_start, x_end - x_start)
    
    img_roi = np.zeros(roi_shape, dtype=np.uint8)
    adjusted_vertices = vertices_np - [x_start, y_start]
    cv2.fillPoly(img_roi, [adjusted_vertices], color=255)
    if cargo_poly:
        adjusteds2 = cargos_np - [x_start, y_start]
        cv2.fillPoly(img_roi, [adjusteds2], color=0)
    
    contours, _ = cv2.findContours(img_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([])
    max_contour = merge_polygons(contours)
    approx = cv2.approxPolyDP(max_contour, epsilon=0.8, closed=True)
    return approx.astype(np.int32) + [x_start, y_start]


def labelme_to_yolo_seg(json_dir, class_map):
    with open(json_dir, "r") as f:
        file_list = [file_path.strip() for file_path in f.readlines()]
    for json_path in file_list:
        json_path = json_path.replace(".jpg", ".json")
        if json_path in ["data/seg-dataset/overflow/250414/sxysw-fjk-tcc-2025041008/cjw-kache-2022111301-190000.json",
                         "data/seg-dataset/overflow/250414/sxysw-fjk-tcc-2025041008/cjw-kache-2022111301-223700.json",
                         "data/seg-dataset/overflow/250414/sxysw-fjk-tcc-2025041008/cjw-kache-2022111301-189975.json",
                         "data/seg-dataset/overflow/250414/sxysw-fjk-tcc-2025041008/cjw-kache-2022111301-159375.json"]:
            continue
        if not json_path.endswith(".json") or not os.path.exists(json_path):
            print(f"{json_path} is error!")
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_width, img_height = data['imageWidth'], data['imageHeight']

        TruckBody_ploys = [shape['points'] for shape in data['shapes'] if shape['label'] == "TruckBody"]
        # Cargo_ploys = [shape['points'] for shape in data['shapes'] if shape['label'] == "Cargo"]
        Cargo_ploys = [shape['points'] for shape in data['shapes'] if shape['label'] in ["Cargo","BuildingMaterials","SandSoil","Cloth"]]
        labels = [shape["label"] for shape in data['shapes'] if shape['label'] in ["Cargo","BuildingMaterials","SandSoil","Cloth"]]
        if len(set(labels)) != len(labels):
            print(json_path, "continue!")
            continue
        Cargo_ploys = [merge_polygons(Cargo_ploys).tolist()] if Cargo_ploys else Cargo_ploys
        yolo_lines = []
        def write_line(normalized, label):
            if len(normalized) >= 6:  # YOLO要求至少3个点（6个坐标值）
                yolo_line = f"{class_map[label]} " + " ".join(map("{:.6f}".format, normalized))
                yolo_lines.append(yolo_line)
            else:
                print(f"无效多边形标注: {json_path} - {label} (点数不足)")
        label = "TruckBody"
        for truckbody in TruckBody_ploys:
            truckbody = np.array(truckbody)
            for cargo in Cargo_ploys:
                if bbox_iou(truckbody, np.array(cargo)) >=0.05:
                    # print("do reduce!")
                    truckbody = get_polygon(truckbody, (img_height, img_width), cargo_poly=cargo)
                    break
            # print(truckbody.reshape((-1,2)).tolist())
            truckbody = (truckbody.reshape((-1,2)) * [1/img_width, 1/img_height]).reshape((-1))
            write_line(truckbody, label)
        label = "Cargo"
        for cargo in Cargo_ploys:
            # print(np.array(cargo).reshape((-1,2)).tolist())
            cargo = (np.array(cargo).reshape((-1,2)) * [1/img_width, 1/img_height]).reshape((-1))
            write_line(cargo, label)
        
        if yolo_lines:
            txt_path = json_path.replace(".json", ".txt")
            with open(txt_path, 'w') as f:
                f.write("\n".join(yolo_lines))
                f.writelines("\n")
        else:
            print(f"无有效标注: {json_path}")


if __name__ == "__main__":
    JSON_DIR = "data/seg-dataset/overflow/train.txt"  # Labelme JSON文件目录
    CLASS_MAP = {"TruckBody": 0, "Cargo": 1}  # 类别映射

    labelme_to_yolo_seg(JSON_DIR, CLASS_MAP)


