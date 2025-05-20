import os
import json
from PIL import Image


def yolo_to_coco(yolo_dir, class_names, task='train'):
    images = []
    annotations = []
    categories = []

    # 创建类别信息
    for i, name in enumerate(class_names):
        categories.append({
            "id": i + 1,
            "name": name,
            "supercategory": "none"
        })

    annotation_id = 1

    # 遍历YOLO标注文件
    with open(os.path.join(yolo_dir, f"{task}.txt"), 'r') as f:
        imgs = [line.strip() for line in f.readlines()]
    # for image_id, filename in enumerate(os.listdir(yolo_dir)):
    for image_id, image_path in enumerate(imgs):
        if not image_path.endswith('.jpg'):
            continue
        filename = image_path.replace("data/det-dataset/person/", "") #os.path.basename(image_path)

        with Image.open(image_path) as img:
            width, height = img.size

        # 添加图像信息
        images.append({
            "id": image_id + 1,
            "file_name": filename,
            "width": width,
            "height": height
        })
        anno_path = image_path.replace('.jpg', '.txt')
        if not os.path.exists(anno_path):
            continue

        # 读取YOLO标注文件
        with open(image_path.replace('.jpg', '.txt'), 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # 转换为COCO格式的边界框 [x_min, y_min, width, height]
            x_min = (x_center - box_width / 2) * width
            y_min = (y_center - box_height / 2) * height
            box_width = box_width * width
            box_height = box_height * height

            # 添加标注信息
            annotations.append({
                "id": annotation_id,
                "image_id": image_id + 1,
                "category_id": class_id + 1,
                "bbox": [x_min, y_min, box_width, box_height],
                "area": box_width * box_height,
                "segmentation": [],
                "iscrowd": 0
            })

            annotation_id += 1

    # 构建COCO格式的JSON
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # 保存为JSON文件
    with open(os.path.join(yolo_dir, f"{task}.json"), 'w') as f:
        json.dump(coco_format, f, indent=4)


if __name__ == "__main__":
    yolo_dir = "data/det-dataset/person"  # YOLO标注文件目录
    class_names = ["person",]  # 类别名称列表

    # yolo_to_coco(yolo_dir, class_names, "train")
    # yolo_to_coco(yolo_dir, class_names, "val")
    yolo_to_coco(yolo_dir, class_names, "test")
