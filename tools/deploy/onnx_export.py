######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: onnx_export.py
# function: convert pth model to onnx model.
######################################################
import argparse
from typing import List, Optional, Tuple

import timm
import torch
from timm import utils
from timm.utils.model import reparameterize_model

from local_lib.models import create_custom_model, FeatExtractModel, MultiLabelModel  # for regster local model

parser = argparse.ArgumentParser(description="PyTorch ImageNet Validation")
parser.add_argument("output", metavar="ONNX_FILE", help="output model filename")
parser.add_argument(
    "--model",
    "-m",
    metavar="MODEL",
    default="mobilenetv3_large_100",
    help="model architecture (default: mobilenetv3_large_100)",
)
parser.add_argument("--opset", type=int, default=12, help="ONNX opset to use (default: 10)")
parser.add_argument(
    "--keep-init",
    action="store_true",
    default=False,
    help="Keep initializers as input. Needed for Caffe2 compatible export in newer PyTorch/ONNX.",
)
parser.add_argument(
    "--aten-fallback",
    action="store_true",
    default=False,
    help="Fallback to ATEN ops. Helps fix AdaptiveAvgPool issue with Caffe2 in newer PyTorch/ONNX.",
)
parser.add_argument(
    "--dynamic-size",
    action="store_true",
    default=False,
    help='Export model width dynamic width/height. Not recommended for "tf" models with SAME padding.',
)
parser.add_argument(
    "--check-forward",
    action="store_true",
    default=False,
    help="Do a full check of torch vs onnx forward after export.",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=1,
    type=int,
    metavar="N",
    help="mini-batch size (default: 1)",
)
parser.add_argument(
    "--img-size",
    default=224,
    type=int,
    metavar="N",
    help="Input image dimension, uses model default if empty",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument("--num-classes", type=int, default=1000, help="Number classes in dataset")
parser.add_argument(
    "--checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="path to checkpoint (default: none)",
)
parser.add_argument("--reparam", default=False, action="store_true", help="Reparameterize model")
parser.add_argument(
    "--use-ema",
    dest="use_ema",
    action="store_true",
    help="use ema version of weights if present",
)
parser.add_argument(
    "--run-test",
    action="store_true",
    help="use test to be sure onnx is same with pth model",
)
parser.add_argument("--model-kwargs", nargs="*", default={}, action=utils.ParseKwargs)
parser.add_argument(
    "--training",
    default=False,
    action="store_true",
    help="Export in training mode (default is eval)",
)
parser.add_argument(
    "--feat-extract-dim",
    default=None,
    type=int,
    help="Feature extraction layer dimension (default: None)",
)
parser.add_argument("--multilabel", default=None, type=dict, help="Multi-label classification")
parser.add_argument("--verbose", default=False, action="store_true", help="Extra stdout output")


def onnx_forward(onnx_file, example_input):
    import onnxruntime

    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(onnx_file, sess_options)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: example_input.numpy()})
    output = output[0]
    return output


def onnx_export(
    model: torch.nn.Module,
    output_file: str,
    example_input: Optional[torch.Tensor] = None,
    training: bool = False,
    verbose: bool = False,
    check: bool = True,
    check_forward: bool = False,
    batch_size: int = 64,
    input_size: Tuple[int, int, int] = None,
    opset: Optional[int] = None,
    dynamic_size: bool = False,
    aten_fallback: bool = False,
    keep_initializers: Optional[bool] = None,
    input_names: List[str] = None,
    output_names: List[str] = None,
):
    import onnx

    if training:
        training_mode = torch.onnx.TrainingMode.TRAINING
        model.train()
    else:
        training_mode = torch.onnx.TrainingMode.EVAL
        model.eval()

    if example_input is None:
        if not input_size:
            assert hasattr(model, "default_cfg")
            input_size = model.default_cfg.get("input_size")
        example_input = torch.randn((batch_size,) + input_size, requires_grad=training)

    # Run model once before export trace, sets padding for models with Conv2dSameExport. This means
    # that the padding for models with Conv2dSameExport (most models with tf_ prefix) is fixed for
    # the input img_size specified in this script.

    # Opset >= 11 should allow for dynamic padding, however I cannot get it to work due to
    # issues in the tracing of the dynamic padding or errors attempting to export the model after jit
    # scripting it (an approach that should work). Perhaps in a future PyTorch or ONNX versions...
    original_out = model(example_input)

    input_names = input_names or ["input0"]
    output_names = output_names or ["output0"]

    dynamic_axes = {"input0": {0: "batch"}, "output0": {0: "batch"}}
    if dynamic_size:
        dynamic_axes["input0"][2] = "height"
        dynamic_axes["input0"][3] = "width"

    if aten_fallback:
        export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    else:
        export_type = torch.onnx.OperatorExportTypes.ONNX

    torch_out = torch.onnx.export(
        model,
        example_input,
        output_file,
        training=training_mode,
        export_params=True,
        verbose=verbose,
        input_names=input_names,
        output_names=output_names,
        keep_initializers_as_inputs=keep_initializers,
        # dynamic_axes=dynamic_axes,
        opset_version=opset,
        operator_export_type=export_type,
    )

    if check:
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model, full_check=True)  # assuming throw on error
        if check_forward and not training:
            import numpy as np

            onnx_out = onnx_forward(output_file, example_input)
            np.testing.assert_almost_equal(torch_out.data.numpy(), onnx_out, decimal=3)
            np.testing.assert_almost_equal(original_out.data.numpy(), torch_out.data.numpy(), decimal=5)


def main():
    args = parser.parse_args()

    args.pretrained = True
    if args.checkpoint:
        args.pretrained = False

    print("==> Creating PyTorch {} model".format(args.model))
    # NOTE exportable=True flag disables autofn/jit scripted activations and uses Conv2dSameExport layers
    # for models using SAME padding
    model = create_custom_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path='', #args.checkpoint,
        exportable=True,
        **args.model_kwargs,
    )

    if args.feat_extract_dim is not None: #feat_extract
        model = FeatExtractModel(model, args.model, args.feat_extract_dim)
    if args.multilabel:
        model = MultiLabelModel(model, args.multilabel)

    timm.models.load_checkpoint(model, args.checkpoint, args.use_ema)

    if args.reparam:
        model = reparameterize_model(model)

    if "redution" in args.model:
        import torch.nn as nn

        if "mobilenetv3" in args.model:
            model.classifier = nn.Identity()  # 移除分类层
        elif "regnet" in args.model:
            model.head.fc = nn.Identity()  # 移除分类层
        else:
            raise f"not support {args.model} !"

    if args.feat_extract_dim is not None:
        model.remove_head() # 只留特征提取层

    onnx_export(
        model,
        args.output,
        opset=args.opset,
        dynamic_size=args.dynamic_size,  # 需要测试改为false是否能不影响转rknn
        aten_fallback=args.aten_fallback,
        keep_initializers=args.keep_init,
        check_forward=args.check_forward,
        training=args.training,
        verbose=args.verbose,
        input_size=(3, args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    def run_test(model, onnx_path, image_path="1.jpg"):
        import cv2
        import numpy as np

        img = cv2.imread(image_path)
        inputs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = cv2.resize(inputs, (224, 224), interpolation=cv2.INTER_CUBIC)
        x = np.array(inputs).astype(np.float32) / 255.0  # ToTensor操作，将像素值范围从[0, 255]转换为[0.0, 1.0]
        x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])  # Normalize操作，使用ImageNet标准进行标准化
        input = x.transpose(2, 0, 1)[np.newaxis, ...]
        input = torch.from_numpy(input).float()  # .to(device)
        output = model(input)
        print("pth", output.detach().numpy())

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = onnxruntime.InferenceSession(onnx_path, sess_options)
        input_name = session.get_inputs()[0].name
        output2 = session.run([], {input_name: [x.transpose(2, 0, 1)]})
        print("onnx", output2)

    if args.run_test:
        # path = "/data/AI-scales/images/0/backflow/00001/1831_8fdaa0cf410f1c36_1673323817187_1673323817536.jpg"
        # from torchvision import transforms
        # from local_lib.data.loader import CustomResize
        # from PIL import Image
        # import numpy as np
        # transform = transforms.Compose([
        #     # transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC, antialias=False),
        #     CustomResize([224, 224]),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        # ])
        # img = Image.open(path)
        # input = transform(img).unsqueeze(0).float()
        # model.eval()
        # output = model(input)
        # print('pth', output.detach().numpy(), path)
        run_test(model, args.output)
    # export onnx (it is recommended to set the opset_version to 12)
    # model.eval()
    # x = torch.randn((args.batch_size, 3, args.img_size, args.img_size))
    # torch.onnx.export(model, x, args.output, opset_version=12, input_names=['input'], output_names=['output'])


if __name__ == "__main__":
    main()
