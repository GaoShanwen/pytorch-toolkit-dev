######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.07.21
# filename: train.py
# function: train pose estimation models using mmpose.
######################################################
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_mmpose(config_path):
    """Delegate to mmpose's train.py."""
    mmpose_train = os.path.join(
        os.path.dirname(__file__), '..', '..', 'mmpose', 'tools', 'train.py')
    mmpose_train = os.path.abspath(mmpose_train)
    if not os.path.exists(mmpose_train):
        raise FileNotFoundError(f"mmpose train.py not found at {mmpose_train}")

    import local_lib.models.symmetry_match.rtmpose_metric  # noqa: F401
    import local_lib.models.symmetry_match.rtmpose_head  # noqa: F401

    import runpy
    import torch
    import warnings
    _torch_load_orig = torch.load
    def _torch_load_patched(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _torch_load_orig(*args, **kwargs)

    torch.load = _torch_load_patched
    sys.path.insert(0, os.path.dirname(mmpose_train))
    runpy.run_path(mmpose_train, run_name='__main__')


if __name__ == "__main__":
    args = sys.argv[1:]
    _run_mmpose(args[0])