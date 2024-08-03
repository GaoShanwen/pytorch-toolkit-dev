import onnxruntime


def onnx_init(input):
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(
        input, sess_options, providers=['AzureExecutionProvider', 'CPUExecutionProvider']
    )
    return session
