import os
import json


categories = ["Truck", "", "Person"]


def get_info(file_path, file_type=".json"):
    if not file_path.endswith(file_type):
        return False
    
    with open(file_path, "r") as file:
        if file_type == ".json":
            return json.load(file)
        return [line.strip().split(" ") for line in file.readlines()]


if __name__=="__main__":
    json_root = "object"
    data_info = {}
    for dir_name in os.listdir(json_root):
        obj_dir = os.path.join(json_root, dir_name)
        for filename in os.listdir(obj_dir):
            data = get_info(os.path.join(obj_dir, filename), ".txt")
            # if not data["shapes"]:
            #     continue
            if not data:
                continue
            # print(data)
            for d in data:
                cat_name = categories[int(d[0])]
                data_info.update({cat_name: data_info.get(cat_name, 0)+1})
    print(data_info)

