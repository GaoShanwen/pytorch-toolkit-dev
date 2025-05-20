import os
import cv2
import json
import yaml
import argparse
import xml.dom.minidom


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate features or SQL queries")
    parser.add_argument('-d', "--obj-root", type=str, default="")
    parser.add_argument('-s', "--src-root", type=str, default="")
    parser.add_argument('-t', "--task", type=str, default="")
    parser.add_argument("--set-cats", type=str, nargs='*', default=None)
    return parser.parse_args()


def analyze_cats(cats):
    if cats is None or len(cats) != 1:
        return cats
    cats = cats[0]
    if cats.endswith(".yaml"):
        with open(cats, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return [v for _, v in data["names"].items()]
    if not (cats.endswith(".txt") or cats.endswith(".names")):
        raise ValueError("Cats should be a TXT file or a list of names.")
    with open(cats, "r") as f:
        return [cat.strip() for cat in f.readlines()]


class XMLWriter(object):
    def __init__(self, cats: list, src_root: str, obj_root: str, anno_type: str='yolo'):
        self.src_root = src_root
        self.obj_root = obj_root
        self.cats = cats
        self.doc = None
        self.data = {}
        self.anno_type = anno_type

    def create_element_for_xml(self, obj_name, obj_value, node):
        node_element = self.doc.createElement(obj_name)
        node_element.appendChild(self.doc.createTextNode(obj_value))
        node.appendChild(node_element)


    def run_write(self, base_name: str, annos: list):
        """
        Converts annotations(YOLO format) to VOC format(Detect Task Type)
        """
        self.doc = xml.dom.minidom.Document()
        root = self.doc.createElement('annotation')
        self.doc.appendChild(root)

        filename = base_name + ".jpg"
        path = os.path.join(self.src_root, filename)
        filename = filename if os.path.exists(path) else base_name + ".png"
        path = os.path.join(self.src_root, filename)

        self.create_element_for_xml("folder", self.src_root, root)
        self.create_element_for_xml("filename", filename, root)
        self.create_element_for_xml("path", path, root)

        sourcename = self.doc.createElement('source')
        self.create_element_for_xml("database", "Unknown", sourcename)
        root.appendChild(sourcename)

        height, width, channel = cv2.imread(path).shape
        nodesize = self.doc.createElement('size')

        self.create_element_for_xml("width", str(width), nodesize)
        self.create_element_for_xml("height", str(height), nodesize)
        self.create_element_for_xml("depth", str(channel), nodesize)
        root.appendChild(nodesize)

        self.create_element_for_xml("segmented", "0", root)
        self.create_element_for_xml("shape_type", "POLYGON", root)
        
        for anno in annos:
            nodeobject = self.doc.createElement('object')
            if self.anno_type == "yolo":
                cat_id, x, y, w, h = map(eval, anno)
                x, y, w, h = width * x, height * y, height * w, height * h
                x1, y1, x2, y2 = list(map(str, map(round, [x-w/2, y-h/2, x+w/2, y+h/2])))
                cat = self.cats[cat_id]
            elif self.anno_type == "coco":
                cat, x1, y1, x2, y2 = anno
            else:
                raise ValueError(f"{self.anno_type} format not supported yet! only support yolo/coco")
            if cat not in self.data:
                self.data.update({cat: 0})
            self.data[cat] += 1
            self.create_element_for_xml("name", cat, nodeobject)
            self.create_element_for_xml("pose", "Unspecified", nodeobject)
            self.create_element_for_xml("truncated", "0", nodeobject)
            self.create_element_for_xml("difficult", "0", nodeobject)

            nodebbox = self.doc.createElement('bndbox')
            self.create_element_for_xml("xmin", str(x1), nodebbox)
            self.create_element_for_xml("ymin", str(y1), nodebbox)
            self.create_element_for_xml("xmax", str(x2), nodebbox)
            self.create_element_for_xml("ymax", str(y2), nodebbox)
            nodeobject.appendChild(nodebbox)
            root.appendChild(nodeobject)

        fp = open(os.path.join(self.obj_root, base_name + '.xml'), 'w', encoding='utf-8')
        self.doc.writexml(fp, indent='  ', newl='\n', addindent='  ')
        fp.close()


class JsonWriter(XMLWriter):
    def __init__(self, cats: list, src_root: str, obj_root: str, anno_type: str='yolo'):
        super().__init__(cats, src_root, obj_root, anno_type)
        assert anno_type != "yolo" or self.cats is not None, "please set cats!"
        self.default_context = {
            "version": "2.4.3",
            "flags": {},
            "shapes": [],
            "imagePath": "",
            "imageData": None,
            "imageHeight": None,
            "imageWidth": None,
            "description": ""
        }

    def run_write(self, base_name: str, annos: list):
        """
        Converts annotations(YOLO format) to COCO format(Detect Task Type)
        """
        if self.anno_type != "yolo":
            raise ValueError(f"{self.anno_type} format not supported yet!")
        this_context = self.default_context
        for img_format in [".jpg", ".png", ".jpeg"]:
            img_path = os.path.join(self.src_root, base_name + img_format)
            if not os.path.exists(img_path):
                continue
            break
        # img_path = os.path.join(self.src_root, base_name + ".png")
        if not os.path.exists(img_path):
            print(f"{img_path} not exist!")
            return
        height, width, _ = cv2.imread(img_path).shape
        shapes = []
        for anno in annos:
            if len(anno) <= 1:
                continue
            assert len(anno) == 5, f"{base_name} file is error!, anno is {anno}"
            cat_id, x, y, w, h = map(eval, anno)
            x, y, w, h = width * x, height * y, width * w, height * h
            x1, y1, x2, y2 = list(map(round, [x-w/2, y-h/2, x+w/2, y+h/2]))
            shapes.append({
                "kie_linking": [],
                "label": self.cats[cat_id],
                "score": None,
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {}
            })
            self.data.update({self.cats[cat_id]: self.data.get(self.cats[cat_id], 0)+1})
        this_context.update({"imageHeight": height})
        this_context.update({"imageWidth": width})
        this_context.update({"imagePath": img_path})
        this_context.update({"shapes": shapes})
        dst_dir = os.path.join(self.obj_root, '/'.join(base_name.split('/')[:-1]))
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        json.dump(this_context, open(os.path.join(self.obj_root, base_name + '.json'), 'w'))
