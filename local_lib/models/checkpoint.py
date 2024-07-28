######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.07.16
# filenaem: checkpoint.py
# function: load and save customized checkpoints.
######################################################
import os
import torch


def filter_inconsistent_channels(state_dict, model):
    return {k: v for k, v in state_dict.items() if k in model.state_dict() \
            and v.shape == model.state_dict()[k].shape}


def load_custom_checkpoint(model, filename, strict=True):
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        state_dict = torch.load(filename)['state_dict']
        state_dict = filter_inconsistent_channels(state_dict, model)
        model.load_state_dict(state_dict, strict=strict)
    else:
        print(f"=> no checkpoint found at '{filename}'")