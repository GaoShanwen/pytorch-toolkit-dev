##################################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.08.03
# filenaem: sys_tools.py
# function: Disable system information output, such as stdout 
# and stderr, during the execution of a function.
##################################################################
import os
import sys


def disable_std_info(sys_names=['stdout','stderr']):
    assert len(sys_names) > 0, "sys_names should not be empty!"
    res_objs = [eval(f"os.dup(sys.{sys_name}.fileno())") for sys_name in sys_names]
    
    with open(os.devnull, 'w') as f:
        for sys_name in sys_names:
            eval(f"os.dup2(f.fileno(), sys.{sys_name}.fileno())")
    return res_objs


def enable_std_info(set_objs, sys_names=['stdout','stderr']):
    assert len(set_objs) == len(sys_names), "set_objs and sys_names should have the same length!"
    for set_obj, sys_name in zip(set_objs, sys_names):
        assert isinstance(set_obj, int), f"set_obj should be an integer, but got {type(set_obj)}"
        eval(f"os.dup2({set_obj}, sys.{sys_name}.fileno())")
        os.close(set_obj)


def hidden_std_info(func):
    def wrapper(*args, **kwargs):
        std_objs = disable_std_info()
        result = func(*args, **kwargs)
        enable_std_info(std_objs)
        return result
    return wrapper

