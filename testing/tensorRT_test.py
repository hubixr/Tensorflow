import tensorrt_dispatch as trt
print("TensorRT version:", trt.__version__)
assert trt.Runtime(trt.Logger())