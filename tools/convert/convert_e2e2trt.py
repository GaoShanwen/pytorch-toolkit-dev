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
    
    # 创建网络定义（TensorRT 8+ 默认使用显式 batch，不需要 EXPLICIT_BATCH 标志）
    network = builder.create_network()
    
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
    
    # 启用 FP16 模式（TensorRT 8+ 会自动根据模型权重选择最优精度）
    # 如果模型是 FP16 导出的，TensorRT 会自动使用 FP16 计算
    # 通过禁用 TF32 来确保更好的 FP16 性能（Ampere+ GPU）
    if fp16_mode:
        if hasattr(builder, 'platform_has_tf32') and builder.platform_has_tf32:
            config.clear_flag(trt.BuilderFlag.TF32)
        print("FP16 mode enabled (TF32 disabled for better FP16 performance)")
    
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
    onnx_path = sys.argv[1]
    fp16 = sys.argv[2] == "fp16"
    engine_path = onnx_path.replace("onnx", "engine")

    build_engine(
        onnx_file_path=onnx_path,
        engine_file_path=engine_path,
        fp16_mode=fp16      # 是否使用 FP16 加速
    )