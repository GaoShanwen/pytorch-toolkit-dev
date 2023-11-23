######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: check_data.py
# function: check owner data before training.
######################################################
import os 
import tqdm
from PIL import Image
import collections


def load_data(anno_path):
    with open(anno_path, 'r') as f:
        lines = [line.strip().split(',') for line in f.readlines() if line.startswith("/data/AI-scales/images")]
    filenames, labels = zip(*(lines))
    return filenames, labels


def load_csv_file(label_file):
    product_id_map = {}
    with open(label_file) as f:
        for line in f:
            # id_record = line.strip().split()
            try:
                id_record = line.strip().replace("'", "").replace("\"", "").split(",")
                product_id_map[id_record[0]] = id_record[1]
            except:
                import pdb; pdb.set_trace()
    return product_id_map


def check_data(filenames):
    for filename in filenames:#tqdm.tqdm(filenames):
        if not os.path.exists(filename):
            print(filename)
        try:  
            # Image.open(filename).verify()  
            with open(filename, 'rb') as f:
                f.seek(-2, 2)
                if not f.read() == b'\xff\xd9':
                    print(filename)
        except IOError:  
            print(filename)


def static_data(train_data, val_data, cat_map):
    # save_cats, _ = zip(*collections.Counter(train_data).most_common())#[:4281])#[:3000])
    # # print(len([id for id, _ in collections.Counter(val_data).items() if id in save_cats]))
    # for cat in save_cats:
    #     print(cat)
    train_counter = collections.Counter(train_data).most_common()#[:999]
    val_dict = dict(collections.Counter(val_data))
    # train_dict = dict(collections.Counter(train_data))
    # val_counter = collections.Counter(val_data).most_common()
    check_dict = val_dict #train_dict
    show_counter = train_counter #val_counter
    
    # import pdb; pdb.set_trace()
    print("| =- cat id -= |  ====----    n a m e    ----====  | =- train -= |  =- val -=  |")
    # train_counter = []
    for id, num1 in show_counter:#.items():
        num2 = check_dict.get(id, '')
        # import pdb; pdb.set_trace()
        # if num1 <= 30:# and isinstance(num2, str):
        #     continue
        id = id.replace(' ', '')
        cat = cat_map[id].split('/')[0]
        print(f"| {id: ^12} | {cat: ^30} | {num1: ^10} | {num2: ^10} |")
    # keys = [name for name in val_dict.keys() if name in train_counter]
    # print(len(keys))


if __name__=="__main__":
    # # load_train_path = "./dataset/exp-data/zero_dataset/train.txt"
    # load_train_path = "./dataset/exp-data/removeredundancy/train.txt"
    # train_files, train_labels = load_data(load_train_path)
    # # check_data(train_files)
    # # load_val_path = "./dataset/exp-data/zero_dataset/val.txt"
    # load_val_path = "./dataset/exp-data/removeredundancy/val.txt"
    # val_files, val_labels = load_data(load_val_path)
    # # check_data(val_files)
    # label_file = "./dataset/exp-data/zero_dataset/label_names.csv"
    # label_map = load_csv_file(label_file)
    # static_data(train_labels, val_labels, label_map)
    import cv2
    import torch
    import numpy as np
    path = "/data/AI-scales/images/0/backflow/00001/1831_8fdaa0cf410f1c36_1673323817187_1673323817536.jpg"
    img_cv2 = cv2.imread(path)
    inputs = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB); inputs = cv2.resize(inputs, (224, 224), interpolation=cv2.INTER_CUBIC)
    x = np.array(inputs).astype(np.float32)/255.  # ToTensor操作，将像素值范围从[0, 255]转换为[0.0, 1.0]  
    x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])  # Normalize操作，使用ImageNet标准进行标准化
    
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    tfl = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC, antialias=False),
        transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),std=torch.tensor([0.229, 0.224, 0.225]))
    ])
    img = Image.open(path)
    import pdb; pdb.set_trace()

    # from torchvision.transforms import InterpolationMode
    # from torchvision.transforms import transforms
    # import torch
    # import cv2
    # SIZE = (224, 224)
    
    # def transform_resize(img, interpolation=InterpolationMode.BILINEAR, antialias=False):
    #     # img = transforms.Resize(SIZE, interpolation, antialias=antialias)(
    #     #     torch.from_numpy(img).permute(2, 0, 1)
    #     # )
    #     img = transforms.Resize([224, 224], interpolation=interpolation, antialias=antialias),
    #     img = img.permute(1, 2, 0).numpy().astype(np.uint8)
    #     return img
    
    # cv_img = cv2.imread(path)
    # cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    # cv_resize = cv2.resize(cv_img_rgb, SIZE, interpolation=cv2.INTER_LINEAR)
    
    # tf_resize_antialias_true = transform_resize(cv_img_rgb, InterpolationMode.BILINEAR, antialias=True)
    # tf_resize_antialias_false = transform_resize(cv_img_rgb, InterpolationMode.BILINEAR, antialias=False)