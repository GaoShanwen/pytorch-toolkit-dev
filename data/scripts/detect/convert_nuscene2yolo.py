import json
import os
import cv2
from tqdm import tqdm

set_categries = {
    "animal": 1, 
    "human.pedestrian.adult": 2, 
    "human.pedestrian.child": 2, 
    "human.pedestrian.construction_worker": 2, 
    "human.pedestrian.personal_mobility": 2, 
    "human.pedestrian.police_officer": 2, 
    "human.pedestrian.stroller": 2, 
    "human.pedestrian.wheelchair": 2, 
    "movable_object.barrier":3, 
    "movable_object.debris":4, 
    "movable_object.pushable_pullable":5, 
    "movable_object.trafficcone":6, 
    "static_object.bicycle_rack":7, 
    "vehicle.bicycle":8, 
    "vehicle.bus.bendy":9, 
    "vehicle.bus.rigid":9, 
    "vehicle.car":10, 
    "vehicle.construction":11, 
    "vehicle.emergency.ambulance":12, 
    "vehicle.emergency.police":13, 
    "vehicle.motorcycle":14, 
    "vehicle.trailer":15, 
    "vehicle.truck":16
}


"""Assistive functions"""
def convert(box,size=(1600,900)):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x, y, w, h]

if __name__ == "__main__":
    anno_path = "data/det-dataset/nuscenes/v1.0-trainval/image_annotations.json"

    with open(anno_path, 'r') as f:
        data = json.load(f)
    for obj in tqdm(data):
        # if obj["filename"] != "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883531412477.jpg":
        #     continue
        file_path = os.path.join("data/det-dataset/nuscenes", obj["filename"]).replace(".jpg", ".txt")
        cat_idx = set_categries[obj["category_name"]]
        line = f"{cat_idx} "
        line += " ".join(map(str, convert(obj["bbox_corners"])))
        with open(file_path, 'a+') as f:
            f.write(line+"\n")
        # print(file_path)
        # break
