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
parser.add_argument("--target-platform", "-tp", default="rk3566", help="eg.: rv1106 (default: rk3566)")
parser.add_argument("--do-quantizate", action="store_true", default=False, help="enable do quantizate")


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
    data_file = "dataset/quantizate/dataset1.txt" if args.do_quantizate else None
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

    path = "dataset/function_test/box_recognize/exp-data/bag/3982_IAIS09B3X22A70341_1677022065935_1677022069543.jpg"
    img = cv2.imread(path)
    inputs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = cv2.resize(inputs, (224, 224), interpolation=cv2.INTER_CUBIC)
    x = np.array(inputs).astype(np.float32) / 255.0  # ToTensor操作，将像素值范围从[0, 255]转换为[0.0, 1.0]
    x = (x - np.array(args.mean)) / np.array(args.std)  # Normalize操作，使用ImageNet标准进行标准化

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(args.input, sess_options, providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output2 = session.run([], {input_name: [x.transpose(2, 0, 1)]})
    import faiss
    np.set_printoptions(suppress=True)
    print("onnx before normalize", output2)
    output2 = np.array(output2[0]).astype(np.float32)
    faiss.normalize_L2(output2)
    print("onnx after normalize", output2)

    output = rknn.inference(inputs=[inputs[np.newaxis, ...]], data_format="nhwc")
    print("rknn before normalize", output)
    output = np.array(output[0]).astype(np.float32)
    faiss.normalize_L2(output)
    print("rknn after normalize", output)
    rknn.release()
