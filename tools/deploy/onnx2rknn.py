######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: onnx2rknn.py
# function: convert onnx model to rknn model.
######################################################
import argparse
import os
import sys
import cv2
import numpy as np
import onnxruntime
from rknn.api import RKNN
import faiss


def args_parse():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Validation")
    parser.add_argument("output", metavar="RKNN_FILE", help="output model filename")
    parser.add_argument("--input", metavar="ONNX_FILE", help="output model filename")
    parser.add_argument('--mean', type=float, nargs='+', default=[0.4850, 0.4560, 0.4060])
    parser.add_argument('--std', type=float, nargs='+', default=[0.2290, 0.2240, 0.2250])
    parser.add_argument("--target-platform", "-tp", default="rk3566", help="eg.: rv1106 (default: rk3566)")
    parser.add_argument("--do-quantizate", action="store_true", default=False, help="enable do quantizate")
    return parser.parse_args()


# 保存当前的stdout
def off_display(sys_name):
    res_obj = eval(f"os.dup(sys.{sys_name}.fileno())")
    with open(os.devnull, 'w') as f:
        eval(f"os.dup2(f.fileno(), sys.{sys_name}.fileno())")
    return res_obj


# 恢复stdout与stderr
def open_display(set_obj, sys_name):
    eval(f"os.dup2({set_obj}, sys.{sys_name}.fileno())")
    os.close(set_obj)


class CustomRKNN(RKNN):
    def rknn_func(self, func_name, return_flag=False, **kwargs):
        ori_stdout, ori_stderr = off_display('stdout'), off_display('stderr')
        
        ret = eval(f"self.{func_name}(**kwargs)")

        # 恢复stdout与stderr
        open_display(ori_stdout, 'stdout')
        open_display(ori_stderr, 'stderr')
        # 打印信息,处理结果
        print(f"run \033[32m{func_name}\033[0m success!")
        if return_flag:
            return ret
        assert ret == 0, f"run {func_name} failed!"


def onnx_init(input):
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(
        input, sess_options, providers=['AzureExecutionProvider', 'CPUExecutionProvider']
    )
    return session


def data_process(img, mean_values, std_values):
    inputs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = cv2.resize(inputs, (224, 224), interpolation=cv2.INTER_CUBIC)
    x = np.array(inputs).astype(np.float32) / 255.0  # ToTensor操作，将像素值范围从[0, 255]转换为[0.0, 1.0]
    x = (x - np.array(mean_values)) / np.array(std_values)  # Normalize操作，使用ImageNet标准进行标准化
    return x, inputs


if __name__ == "__main__":
    args = args_parse()
    # Create RKNN object
    rknn = CustomRKNN(verbose=True)
    mean_values = [num * 255.0 for num in args.mean]
    std_values = [num * 255.0 for num in args.std]
    data_file = "dataset/quantizate/dataset1.txt" if args.do_quantizate else None

    rknn.rknn_func("config", mean_values=mean_values, std_values=std_values, target_platform=args.target_platform)
    rknn.rknn_func("load_onnx", model=args.input)
    rknn.rknn_func("build", do_quantization=args.do_quantizate, dataset=data_file)
    rknn.rknn_func("export_rknn", export_path=args.output)
    rknn.rknn_func("init_runtime")

    path = "dataset/function_test/box_recognize/exp-data/bag/3982_IAIS09B3X22A70341_1677022065935_1677022069543.jpg"
    x, inputs = data_process(cv2.imread(path), args.mean, args.std)
    # print("x: ", x[0,:6,:])

    session = onnx_init(args.input)
    output2 = session.run([], {session.get_inputs()[0].name: [x.transpose(2, 0, 1)]})
    output = rknn.rknn_func("inference", return_flag=True, inputs=[inputs[np.newaxis, ...]], data_format="nhwc")
    # 打印ONNX和RKNN模型的前N个结果，并对其进行归一化操作
    np.set_printoptions(suppress=True) # 取消科学计数法输出
    def print_out(name, out, number=5):
        out = np.array(out[0]).astype(np.float32)
        print(f"{name} before normalize", out[:, :number])
        faiss.normalize_L2(out)
        print(f"{name} after normalize", out[:, :number])

    print_out("onnx", output2)
    print_out("rknn", output)

    # 释放资源
    rknn.rknn_func("release", return_flag=True)
    _, _ = off_display('stdout'), off_display('stderr')
