import tensorrt as trt
import os
import numpy as np


def build_engine(onnx_file_path, engine_file_path, fp16_mode=False, max_workspace_size=4,
                 min_batch=1, opt_batch=4, max_batch=8):
    """
    使用 TensorRT API 将 ONNX 转换为 Engine

    Args:
        onnx_file_path: ONNX 模型路径
        engine_file_path: 输出的 engine 文件路径
        fp16_mode: 是否使用 FP16 精度, 无效，TRT11 自动选择精度
        max_workspace_size: 最大工作空间大小（GB）
        min_batch: 最小 batch size
        opt_batch: 最优 batch size
        max_batch: 最大 batch size
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network = builder.create_network()

    parser = trt.OnnxParser(network, logger)

    print(f"Loading ONNX file from path {onnx_file_path}...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print(f"ONNX file loaded successfully. Building TensorRT engine...")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        max_workspace_size * (1 << 30)
    )

    if fp16_mode:
        print("FP16 mode requested (TensorRT 11 auto-selects precision)")

    num_inputs = network.num_inputs
    if num_inputs > 0:
        profile = builder.create_optimization_profile()
        for i in range(num_inputs):
            tensor_name = network.get_input(i).name
            shape = network.get_input(i).shape

            min_shape = []
            opt_shape = []
            max_shape = []
            for j, dim in enumerate(shape):
                if dim == -1 or isinstance(dim, str):
                    min_shape.append(1 if j == 0 else dim)
                    opt_shape.append(opt_batch if j == 0 else dim)
                    max_shape.append(max_batch if j == 0 else dim)
                else:
                    min_shape.append(int(dim))
                    opt_shape.append(int(dim))
                    max_shape.append(int(dim))

            profile.set_shape(tensor_name, min_shape, opt_shape, max_shape)
            print(f"  Input '{tensor_name}': min={min_shape}, opt={opt_shape}, max={max_shape}")

        config.add_optimization_profile(profile)

    print("Building engine...  This may take a few minutes.")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build engine")
        return None

    print(f"Saving engine to {engine_file_path}")
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)

    print("Engine built and saved successfully!")
    return engine_file_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorRT engine')
    parser.add_argument('onnx', type=str, help='Path to ONNX file')
    parser.add_argument('--engine', type=str, default=None, help='Path to output engine file')
    parser.add_argument('--fp16', action='store_true', default=True, help='Enable FP16 mode')
    parser.add_argument('--workspace', type=int, default=8, help='Workspace size in GB')
    args = parser.parse_args()

    onnx_path = args.onnx
    engine_path = args.engine or onnx_path.replace('.onnx', '.engine')

    build_engine(
        onnx_file_path=onnx_path,
        engine_file_path=engine_path,
        fp16_mode=args.fp16,
        max_workspace_size=args.workspace
    )