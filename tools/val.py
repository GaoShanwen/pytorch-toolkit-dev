######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.07.29
# filenaem: val.py
# function: validate dataset use yolo or mmpose.
#   - mmpose mode: first arg is a .py config file, delegates to mmpose tools/test.py
#   - yolo mode:   uses ultralytics YOLO validator
######################################################
import sys
import os
import runpy
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_mmpose(config_path):
    """Delegate to mmpose's test.py."""
    mmpose_test = os.path.join(
        os.path.dirname(__file__), '..', '..', 'mmpose', 'tools', 'test.py')
    mmpose_test = os.path.abspath(mmpose_test)
    if not os.path.exists(mmpose_test):
        raise FileNotFoundError(f"mmpose test.py not found at {mmpose_test}")

    # Register custom local_lib modules (metric) before loading config
    import local_lib.models.symmetry_match.rtmpose_head  # noqa: F401
    import local_lib.models.symmetry_match.rtmpose_metric  # noqa: F401
    import local_lib.visualization.custom_visualizer  # noqa: F401
    import local_lib.data.mix_dataset.coco_merge  # noqa: F401

    _torch_load_orig = torch.load
    def _torch_load_patched(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _torch_load_orig(*args, **kwargs)
    torch.load = _torch_load_patched
    sys.path.insert(0, os.path.dirname(mmpose_test))
    runpy.run_path(mmpose_test, run_name='__main__')


if __name__ == "__main__":
    args = sys.argv[1:]
    _run_mmpose(args[0])