######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: onnx2rknn.py
# function: convert onnx model to rknn model.
######################################################
import argparse
import cv2
import onnxruntime
import numpy as np
from rknn.api import RKNN


parser = argparse.ArgumentParser(description="PyTorch ImageNet Validation")
parser.add_argument("output", metavar="RKNN_FILE", help="output model filename")
parser.add_argument("--input", metavar="ONNX_FILE", help="output model filename")
parser.add_argument("--target-platform", "-tp", default="rk3566", help="target platform (default: rk3566)")
parser.add_argument("--do-quantizate", action="store_true", default=False, help="enable do quantizate")


if __name__ == "__main__":
    args = parser.parse_args()
    # Create RKNN object
    rknn = RKNN(verbose=True)
    mean_vales = [num * 255.0 for num in args.mean]
    std_values = [num * 255.0 for num in args.std]
    rknn.config(mean_values=mean_vales, std_values=std_values, target_platform=args.target_platform)
    print("Config model done")

    print("--> Loading model")
    onnx_model = args.input  # "output/converted_model/20230925-185645-resnet18-224.onnx"
    ret = rknn.load_onnx(
        model=onnx_model
    )  # , inputs=["input0"], input_size_list=[[1, args.in_chans, args.img_size, args.img_size]])
    if ret != 0:
        print("load model failed!")
        exit(ret)
    print("done")

    # Build model
    print("--> Building model")
    data_file = "dataset/minidata/quantizate/dataset1.txt" if args.do_quantizate else None
    ret = rknn.build(do_quantization=args.do_quantizate, dataset=data_file)  # , rknn_batch_size=args.batch_size)
    # rknn.accuracy_analysis(inputs=['dataset/.../image.jpg'])
    if ret != 0:
        print("build model failed.")
        exit(ret)
    print("done")

    # Export rknn model
    print("--> Export RKNN model")
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print("Export rknn failed!")
        exit(ret)
    print("Export rknn success!")

    # # Set inputs
    # img_root = "dataset/minidata/validation"
    # evaliation(img_root, rknn, onnx_model)
    ret = rknn.init_runtime()
    if ret != 0:
        print("init rknn failed!")
        exit(ret)
    # feat_extract(args, rknn)

    # img=cv2.imread("dataset/minidata/quantizate/100_NZ53MZV0KS_1680344371005_1680344371719.jpg")
    path = "/data/AI-scales/images/0/backflow/00001/1831_8fdaa0cf410f1c36_1673323817187_1673323817536.jpg"
    # path = "./dataset/1.jpg"
    img = cv2.imread(path)
    inputs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = cv2.resize(inputs, (224, 224), interpolation=cv2.INTER_CUBIC)
    x = np.array(inputs).astype(np.float32) / 255.0  # ToTensor操作，将像素值范围从[0, 255]转换为[0.0, 1.0]
    x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])  # Normalize操作，使用ImageNet标准进行标准化

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(args.input, sess_options)
    input_name = session.get_inputs()[0].name
    output2 = session.run([], {input_name: [x.transpose(2, 0, 1)]})
    print("onnx", output2)

    output = rknn.inference(inputs=[inputs[np.newaxis, ...]], data_format="nhwc")
    print("rknn", output)
    rknn.release()
