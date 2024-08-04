######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: onnx2rknn.py
# function: convert onnx model to rknn model.
######################################################
import argparse
import cv2
import faiss
import numpy as np

import logging
from colorama import Fore, Style

from rknn.api import RKNN

from local_lib.utils import hidden_std_info, disable_std_info, onnx_init
from local_lib.data.loader import data_process

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(f"[{Fore.MAGENTA} onnx2rknn {Style.RESET_ALL}]")


def args_parse():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Validation")
    parser.add_argument("output", metavar="RKNN_FILE", help="output model filename")
    parser.add_argument("--input", metavar="ONNX_FILE", help="output model filename")
    parser.add_argument('--mean', type=float, nargs='+', default=[0.4850, 0.4560, 0.4060])
    parser.add_argument('--std', type=float, nargs='+', default=[0.2290, 0.2240, 0.2250])
    parser.add_argument("--debug", action="store_true", default=False, help="enable debug mode")
    parser.add_argument("--target-platform", "-tp", default="rk3566", help="eg.: rv1106 (default: rk3566)")
    parser.add_argument("--do-quantizate", action="store_true", default=False, help="enable do quantizate")
    return parser.parse_args()


class CustomRKNN(RKNN):
    def __init__(self, debug=False, *args, **kwargs):
        super(CustomRKNN, self).__init__(*args, **kwargs)
        self.custom_func = self.normal_run if debug else self.run_func

    @hidden_std_info
    def run_func(self, func, **kwargs):
        return eval(f"self.{func}(**kwargs)")
    
    def normal_run(self, func, **kwargs):
        return eval(f"self.{func}(**kwargs)")
    
    def rknn_func(self, func_name, check_flag=True, **kwargs):
        ret = self.custom_func(func_name, **kwargs)
        if not check_flag:
            return ret
        check_res = f"{Fore.GREEN}success" if ret == 0 else f"{Fore.RED}failure"
        _logger.info(f"run {Fore.BLUE}{func_name}{Style.RESET_ALL} {check_res}{Style.RESET_ALL}!")


def print_out(name, out, number=5):
    out = np.array(out[0]).astype(np.float32)
    prefix = f"{Fore.BLUE}{name}{Style.RESET_ALL}"
    _logger.info(f"{prefix} original   out:{Fore.YELLOW}{out[:, :number]}{Style.RESET_ALL}")
    faiss.normalize_L2(out)
    _logger.info(f"{prefix} normalized out:{Fore.YELLOW}{out[:, :number]}{Style.RESET_ALL}")


if __name__ == "__main__":
    args = args_parse()
    # Create RKNN object
    rknn = CustomRKNN(debug=args.debug, verbose=True)
    mean_values = [num * 255.0 for num in args.mean]
    std_values = [num * 255.0 for num in args.std]
    data_file = "dataset/quantizate/dataset1.txt" if args.do_quantizate else None

    rknn.rknn_func("config", mean_values=mean_values, std_values=std_values, target_platform=args.target_platform)
    rknn.rknn_func("load_onnx", model=args.input)
    rknn.rknn_func("build", do_quantization=args.do_quantizate, dataset=data_file)
    rknn.rknn_func("export_rknn", export_path=args.output)
    rknn.rknn_func("init_runtime")

    path = "dataset/function_test/box_recognize/exp-data/bag/3982_IAIS09B3X22A70341_1677022065935_1677022069543.jpg"
    img = cv2.imread(path)
    x, inputs = data_process(img, args.mean, args.std)
    np.set_printoptions(suppress=True) # 取消科学计数法输出
    _logger.info(f"original   img: \n{Fore.YELLOW}{img[0, :6]}{Style.RESET_ALL}")
    _logger.info(f"normalized img: \n{Fore.YELLOW}{x[0, :6]}{Style.RESET_ALL}")

    session = onnx_init(args.input)
    output2 = session.run([], {session.get_inputs()[0].name: [x.transpose(2, 0, 1)]})
    output = rknn.rknn_func("inference", check_flag=False, inputs=[inputs[np.newaxis, ...]], data_format="nhwc")

    print_out("onnx", output2)
    print_out("rknn", output)

    rknn.rknn_func("release", check_flag=False)
    disable_std_info()
