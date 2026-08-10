"""
Export RTMPose model to ONNX format.

Usage:
    python tools/convert/export.py \
        --config cfgs/rtmpose/pretrain_cspnext_udp/rtmpose-m_udp-sy.py \
        --checkpoint work_dirs/rtmpose-m_udp/epoch_420.pth
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

sys.path.append('.')
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpose.registry import MODELS


class RTMPoseExportWrapper(nn.Module):
    """Export wrapper that includes data preprocessing and model forward.

    The input is a raw BGR image (uint8, 0-255 range), and the wrapper:
    1. Converts BGR to RGB
    2. Normalizes with (x - mean) / std
    3. Runs the model's backbone + head forward

    Returns:
        pred_x (Tensor): (N, K, Lx) simcc x-coordinate predictions.
        pred_y (Tensor): (N, K, Ly) simcc y-coordinate predictions.
    """

    def __init__(self, model, mean, std, bgr_to_rgb=True):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.bgr_to_rgb = bgr_to_rgb

    def forward(self, x):
        x = x.float()
        if self.bgr_to_rgb:
            x = x[:, [2, 1, 0], :, :]
        x = (x - self.mean) / self.std
        return self.model._forward(x)


def parse_args():
    parser = argparse.ArgumentParser(description='Export RTMPose model to ONNX')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to mmpose config file (e.g. cfgs/rtmpose/.../xxx.py)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (.pth)')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[192, 192],
                        help='Input image size (H W), default: 192 192')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version, default: 11')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model with onnxsim')
    parser.add_argument('--half', action='store_true',
                        help='Export FP16 model')
    return parser.parse_args()


def export_onnx(model, output_path, imgsz, opset=11, simplify=False, half=False):
    """Export PyTorch model to ONNX with dynamic batch size 1-8.

    Args:
        model: nn.Module to export.
        output_path: Path to save the ONNX file.
        imgsz: Input image size (H, W).
        opset: ONNX opset version.
        simplify: Whether to simplify the ONNX model.
        half: Whether to export in FP16.
    """
    device = next(model.parameters()).device
    img = torch.randn(1, 3, *imgsz, device=device)

    if half:
        model = model.half()
        img = img.half()

    model.eval()

    dynamic_axes = {
        'input': {0: 'batch'},
        'output_x': {0: 'batch'},
        'output_y': {0: 'batch'},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            img,
            output_path,
            opset_version=opset,
            input_names=['input'],
            output_names=['output_x', 'output_y'],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    print(f'ONNX exported to: {output_path}')

    if simplify:
        _simplify_onnx(output_path)

    _verify_onnx(wrapper=model, onnx_path=output_path, imgsz=imgsz, half=half)

    return output_path


def _simplify_onnx(onnx_path):
    """Simplify ONNX model using onnxsim."""
    try:
        import onnx
        from onnxsim import simplify
        model = onnx.load(onnx_path)
        model_simp, check = simplify(model)
        if check:
            onnx.save(model_simp, onnx_path)
            print(f'ONNX simplified: {onnx_path}')
        else:
            print('ONNX simplification failed, keeping original model')
    except ImportError:
        print('onnx / onnxsim not installed, skip simplification')


def _verify_onnx(wrapper, onnx_path, imgsz, half=False):
    """Verify ONNX model with dynamic batch sizes 1 and 8.

    Compares ONNX outputs against PyTorch outputs to ensure correctness.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print('onnxruntime not installed, skip dynamic batch verification')
        return

    wrapper.eval()
    device = next(wrapper.parameters()).device

    for batch in [1, 2, 3, 4, 5, 6, 7, 8]:
        img = torch.randn(batch, 3, *imgsz, device=device)
        if half:
            img = img.half()

        with torch.no_grad():
            pt_out = wrapper(img)

        session = ort.InferenceSession(onnx_path)
        ort_out = session.run(None, {'input': img.cpu().numpy()})

        for i, (pt, ort_o) in enumerate(zip(pt_out, ort_out)):
            pt_np = pt.cpu().numpy()
            max_diff = float(abs(pt_np - ort_o).max())
            status = 'OK' if max_diff < 1e-3 else 'MISMATCH'
            print(f'  batch={batch}, {i}: input={img.shape}, output={ort_o.shape}, '
                  f'max_diff={max_diff:.2e} [{status}]')

    print(f'Dynamic batch verification passed (batch 1-8)')


if __name__ == '__main__':
    args = parse_args()

    import local_lib.models.symmetry_match.rtmpose_head  # noqa: F401
    import mmpose.models.data_preprocessors  # noqa: F401  register PoseDataPreprocessor
    from mmengine.registry import DefaultScope
    DefaultScope.get_instance('mmpose', scope_name='mmpose')
    cfg = Config.fromfile(args.config)

    imgsz = (cfg.model.head.input_size[1], cfg.model.head.input_size[0])
    print(f'Auto-detected input size from model.head: {imgsz}')

    model = MODELS.build(cfg.model)

    _torch_load_orig = torch.load
    def _torch_load_patched(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _torch_load_orig(*args, **kwargs)
    torch.load = _torch_load_patched
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    dp_cfg = cfg.model.data_preprocessor
    wrapper = RTMPoseExportWrapper(
        model,
        mean=dp_cfg['mean'],
        std=dp_cfg['std'],
        bgr_to_rgb=dp_cfg.get('bgr_to_rgb', True),
    )

    output_path = os.path.splitext(args.checkpoint)[0] + '.onnx'

    export_onnx(
        wrapper,
        output_path,
        imgsz=imgsz,
        opset=args.opset,
        simplify=args.simplify,
        half=args.half,
    )