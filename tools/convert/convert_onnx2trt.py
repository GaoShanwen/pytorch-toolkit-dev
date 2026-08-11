import tensorrt as trt
import os
import sys

def build_engine(onnx_file_path, engine_file_path, fp16_mode=True, max_workspace_size=4):
    """
    使用 TensorRT API 将 ONNX 转换为 Engine
    
    Args:
        onnx_file_path: ONNX 模型路径
        engine_file_path: 输出的 engine 文件路径
        fp16_mode: 是否使用 FP16 精度（Orin Nano 支持，速度更快）
        max_workspace_size: 最大工作空间大小（GB）
    """
    # 创建 logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # 创建 builder
    builder = trt.Builder(logger)
    
    # 创建网络定义（显式 batch）
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    
    # 创建 ONNX 解析器
    parser = trt.OnnxParser(network, logger)
    
    # 解析 ONNX 文件
    print(f"Loading ONNX file from path {onnx_file_path}...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print(f"ONNX file loaded successfully. Building TensorRT engine...")
    
    # 创建 builder 配置
    config = builder.create_builder_config()
    
    # 设置工作空间大小
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 
        max_workspace_size * (1 << 30)  # 转换为字节
    )
    
    # 启用 FP16 模式（Orin Nano 支持）
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag. FP16)
        print("FP16 mode enabled")
    
    # 构建 engine
    print("Building engine...  This may take a few minutes.")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Failed to build engine")
        return None
    
    # 保存 engine 到文件
    print(f"Saving engine to {engine_file_path}")
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    
    print("Engine built and saved successfully!")
    return engine_file_path


# 使用示例
if __name__ == "__main__":
    suffix = sys.argv[1] if len(sys.argv) > 1 else ''
    assert suffix != '' and len(suffix) == 4, "suffix must be MMDD, e.g. 0811"
    onnx_path = f"models/onnx/yolo26s-pose-c15k7v26{suffix}.onnx"
    engine_path = onnx_path.replace("onnx", "engine")

    build_engine(
        onnx_file_path=onnx_path,
        engine_file_path=engine_path,
        fp16_mode=True
    )