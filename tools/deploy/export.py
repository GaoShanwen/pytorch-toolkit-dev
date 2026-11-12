import argparse
import torch
import sys
import os
sys.path.append('.')
from local_lib.utils import load_config
from local_lib.models import build_model_from_checkpoint


if __name__=="__main__":
    obj_dir = sys.argv[1]
    model_path = os.path.join(obj_dir, "last.pth")
    cfg_path = os.path.join(obj_dir, "config.yaml")
    cfg = load_config(cfg_path)
    classes = cfg["classes"]

    # Load a pretrained YOLO26n model 
    model = build_model_from_checkpoint(cfg, model_path, "cpu", len(classes))
    model.eval()

    # Export the model to ONNX format for deployment

    dynamic_axes = {0: "batch_size", 2: "height", 3: "width"}
    # onnx_model_name = model_path.replace(".pth", ".onnx")
    # onnx_model = torch.onnx.export(
    #     model,  # model being run
    #     torch.randn(1, 3, 128, 128),  # model input
    #     onnx_model_name,  # where to save the model (can be a file or file-like object)
    #     export_params=True,  # store the trained parameter weights inside the model file
    #     opset_version=17,  # the ONNX version to export
    #     do_constant_folding=True,  # whether to execute constant folding for optimization
    #     input_names=["input"],  # the model's input names
    #     output_names=["output"],  # the model's output names
    #     dynamic_axes={  # variable length axes
    #         "input": dynamic_axes,
    #         "output": dynamic_axes,
    #     },
    # )
    onnx_program = torch.onnx.dynamo_export(model, torch.randn(1, 3, 128, 128))
    onnx_program.save(model_path.replace(".pth", ".onnx"))
    # model.to_onnx(model_path.replace(".pth", ".onnx"))
    # with torch.no_grad(): # 必须关闭梯度
    #     path = model.export(format="onnx", imgsz=[128, 128])  # Returns the path to the exported model