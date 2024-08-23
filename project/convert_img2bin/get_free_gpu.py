######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2024.07.05
# filenaem: get_free_gpu.py
# function: get the free gpu id.
######################################################
import argparse
import time
import os
from py3nvml import py3nvml
from py3nvml.utils import try_get_info


def get_free_gpuid(start_id, end_id, check_root):
    # 检查每个GPU是否为空闲
    for i in range(start_id, end_id):
        check_file = os.path.join(check_root, str(i))
        if os.path.exists(check_file):
            print(f"GPU {i} was busy")
            continue  # 该GPU已在运行，跳过
        # 获取GPU的handle
        handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
        procs = try_get_info(py3nvml.nvmlDeviceGetComputeRunningProcesses, handle, ['something'])
        if not procs:
            open(check_file, 'w').close()  # 该GPU为空闲，创建文件
            print(f"GPU {i} is free")
            return i  # 找到一个空闲GPU，返回其ID
        print(f"GPU {i} is running")
    return -1  # 所有GPU都在运行，返回-1
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get free gpus id from set of devices')
    parser.add_argument('-s', '--start-point', type=int, default=0, help='the start point of gpu id')
    parser.add_argument('-e', '--end-point', type=int, default=8, help='the end point of gpu id')
    parser.add_argument('-t', '--sleep-time', type=int, default=20, help='the wait time of checking gpu status')
    parser.add_argument('-r', '--flag-root', type=str, help='the gpu whether is run')
    args = parser.parse_args()
    sleep_time = args.sleep_time

    # 初始化NVML
    py3nvml.nvmlInit()
    # # 获取GPU的数量
    # gpu_num = py3nvml.nvmlDeviceGetCount()
    while True:
        check_result = get_free_gpuid(args.start_point, args.end_point, args.flag_root)
        if check_result != -1:
            print(f"true {check_result}")
            break  # 找到一个空闲GPU，跳出循环
        else:
            print(f"false")
            time.sleep(sleep_time)  # 等待sleep_time秒后再次检查
    # # 关闭NVML
    # py3nvml.nvmlShutdown()