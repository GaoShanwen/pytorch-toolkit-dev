######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.07.29
# filenaem: predict.py
# function: predict dataset use yolo or mmpose.
#   - mmpose mode: first arg is a .py config file, delegates to mmpose demo/image_demo.py
#   - yolo mode:   uses ultralytics YOLO predictor
######################################################
import sys
import os


def _run_mmpose(config_path):
    """Delegate to mmpose's demo/image_demo.py."""
    mmpose_demo = os.path.join(
        os.path.dirname(__file__), '..', '..', 'mmpose', 'demo', 'image_demo.py')
    mmpose_demo = os.path.abspath(mmpose_demo)
    if not os.path.exists(mmpose_demo):
        raise FileNotFoundError(f"mmpose image_demo.py not found at {mmpose_demo}")
    import runpy
    import torch
    _torch_load_orig = torch.load
    def _torch_load_patched(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _torch_load_orig(*args, **kwargs)
    torch.load = _torch_load_patched
    sys.path.insert(0, os.path.dirname(mmpose_demo))
    runpy.run_path(mmpose_demo, run_name='__main__')


if __name__ == "__main__":
    args = sys.argv[1:]
    _run_mmpose(args[0])