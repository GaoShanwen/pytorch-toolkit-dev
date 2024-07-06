######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.07.05
# filenaem: get_free_gpu.py
# function: get the free gpu id.
######################################################
from py3nvml import py3nvml
from py3nvml.utils import try_get_info


def get_free_gpuid(gpu_count):
    # 检查每个GPU是否为空闲
    for i in range(gpu_count):
        # 获取GPU的handle
        handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
        procs = try_get_info(py3nvml.nvmlDeviceGetComputeRunningProcesses, handle, ['something'])
        if not procs:
            return i  # 找到一个空闲GPU，返回其ID
    return -1  # 所有GPU都在运行，返回-1
    

if __name__ == '__main__':
    # 初始化NVML
    py3nvml.nvmlInit()
    # 获取GPU的数量
    gpu_num = py3nvml.nvmlDeviceGetCount()
    while True:
        check_result = get_free_gpuid(gpu_num)
        if check_result != -1:
            print(f"true {check_result}")
            break  # 找到一个空闲GPU，跳出循环
        else:
            print(f"false")
            time.sleep(20000)  # 等待20秒后再次检查
    # 关闭NVML
    py3nvml.nvmlShutdown()