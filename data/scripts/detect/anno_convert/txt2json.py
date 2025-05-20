import os
from tqdm import tqdm

from tools import parse_args, analyze_cats, JsonWriter


class Txt2Json(object):
    def __init__(self, cfg, **kwargs):
        self.src_root = cfg.src_root
        self.obj_root = cfg.obj_root or cfg.src_root
        anno_type = kwargs.get("anno_type", "yolo")
        if cfg.set_cats is None:
            cfg.set_cats = [os.path.join(cfg.src_root, "dataset.yaml")]
        self.cats = analyze_cats(cfg.set_cats)
        print(self.cats)
        
        self.writer = JsonWriter(self.cats, self.src_root, self.obj_root, anno_type)
        self.convert2json()
        print(f"Convert complete, Every object number: {self.writer.data}")
        try:
            print(f"Convert complete, Total object number: {sum(self.writer.data)}")
        except Exception as e:
            print(f"Convert complete, Total object number: {sum(self.writer.data.values())}")
    
    def convert2json(self):
        filelist = [
            os.path.join(root.replace(self.src_root+'/', ''), file) \
            for root, dirs, files in os.walk(self.src_root) \
                for file in files if root != self.src_root and file.endswith(".txt") \
        ]
        # filelist = [filename for filename in os.listdir(self.src_root) if filename.endswith(".txt")]
        for file_name in tqdm(filelist):
            base_name = os.path.splitext(file_name)[0]
            with open(os.path.join(self.src_root, base_name + ".txt")) as f:
                annos = [line.strip().split(' ') for line in f.readlines()]
            self.writer.run_write(base_name, annos)

if __name__ == '__main__':
    args = parse_args()
    Txt2Json(args)
