######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: onnx2rknn.py
# function: convert onnx model to rknn model.
######################################################
import argparse
import os
import cv2
import numpy as np
import onnxruntime
from rknn.api import RKNN

parser = argparse.ArgumentParser(description="PyTorch ImageNet Validation")
parser.add_argument("output", metavar="RKNN_FILE", help="output model filename")
parser.add_argument("--input", metavar="ONNX_FILE", help="output model filename")
parser.add_argument('--mean', type=float, nargs='+', default=[0.4850, 0.4560, 0.4060])
parser.add_argument('--std', type=float,  nargs='+', default=[0.2290, 0.2240, 0.2250])
parser.add_argument("--target-platform", "-tp", default="rk3566", help="target platform (default: rk3566)")
parser.add_argument("--do-quantizate", action="store_true", default=False, help="enable do quantizate")


def evaliation(img_root, rknn):
    top1_rknn, top1_onnx = 0, 0
    resize_width, resize_height = 224, 224 
    img_num = 0

    init_feats_dir(args.results_dir)
    pbar = tqdm.tqdm(total=len())
    for cat_index, cat in enumerate(sorted(os.listdir(img_root))):
        img_dir = os.path.join(img_root, cat)
        cat_rknn, cat_onnx = 0, 0
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
            
            input = resized_image[np.newaxis, ...]
            
            rknn_output = rknn.inference(inputs=[input.astype(np.float32)])
            # save_feat(rknn_output[0], target, batch_idx, args.results_dir)
            x = rknn_output[0]; output = np.exp(x)/np.sum(np.exp(x)); rknn_idx = np.argmax(output)
            cat_rknn += rknn_idx == cat_index
            img_num += 1
            pbar.update(1)
        top1_rknn += cat_rknn
        print(f"{cat}-onnx_idx: {cat_onnx/len(os.listdir(img_dir))} | rknn_idx: {cat_rknn/len(os.listdir(img_dir))}")
    pbar.close()
    print(f"onnx_top1: {top1_onnx}/{img_num}|{top1_onnx/img_num} | rknn_top1: {top1_rknn}/{img_num}|{top1_rknn/img_num}")


if __name__ == "__main__":
    args = parser.parse_args()
    # Create RKNN object
    rknn = RKNN(verbose=True)
    mean_values = [num * 255.0 for num in args.mean]
    std_values = [num * 255.0 for num in args.std]
    rknn.config(mean_values=mean_values, std_values=std_values, target_platform=args.target_platform)
    ret = rknn.load_onnx(model=args.input)
    assert ret == 0, "load model failed!"
    print("done")

    # Build model
    print("--> Building model")
    data_file = "dataset/minidata/quantizate/dataset1.txt" if args.do_quantizate else None
    ret = rknn.build(do_quantization=args.do_quantizate, dataset=data_file)  # , rknn_batch_size=args.batch_size)
    # rknn.accuracy_analysis(inputs=['dataset/.../image.jpg'])
    assert ret == 0, "build model failed!"
    print("done")

    # Export rknn model
    print("--> Export RKNN model")
    ret = rknn.export_rknn(args.output)
    assert ret == 0, "Export rknn failed!"
    print("Export rknn success!")

    ret = rknn.init_runtime()
    assert ret == 0, "init rknn failed!"

    # assert args.input_mode in ["path", "dir", "file"], "please set infer_mode to path, dir, or files"
    # if args.input_mode == "file":
    #     with open(args.data_path, "r") as f:
    #         query_files = ([line.strip("\n").split(",")[0] for line in f.readlines()])
    # else:
    #     query_files = (
    #         [os.path.join(args.data_path, path) for path in os.listdir(args.data_path)]
    #         if args.input_mode == "dir"
    #         else [args.data_path]
    #     )
    # img=cv2.imread("dataset/minidata/quantizate/100_NZ53MZV0KS_1680344371005_1680344371719.jpg")
    # path = "/data/AI-scales/images/0/backflow/00001/1831_8fdaa0cf410f1c36_1673323817187_1673323817536.jpg"
    path = "./dataset/test_imgs/1.jpg"
    img = cv2.imread(path)
    inputs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = cv2.resize(inputs, (224, 224), interpolation=cv2.INTER_CUBIC)
    x = np.array(inputs).astype(np.float32) / 255.0  # ToTensor操作，将像素值范围从[0, 255]转换为[0.0, 1.0]
    x = (x - np.array(args.mean)) / np.array(args.std)  # Normalize操作，使用ImageNet标准进行标准化

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(args.input, sess_options)
    input_name = session.get_inputs()[0].name
    output2 = session.run([], {input_name: [x.transpose(2, 0, 1)]})
    print("onnx", output2)

    output = rknn.inference(inputs=[inputs[np.newaxis, ...]], data_format="nhwc")
    print("rknn", output)
    rknn.release()
