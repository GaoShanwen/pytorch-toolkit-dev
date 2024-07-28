from ctypes import cdll, c_char_p, c_int, c_float
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 加载.so库
lib = cdll.LoadLibrary('project/predictor4linux/build/libsmart_predictor.so')

# 定义函数参数类型
lib.predictInit.argtypes = [c_char_p, c_char_p, c_int]
lib.predict.argtypes = [c_char_p, c_float]
lib.predict.restype = c_char_p

# 准备传递给C++函数的字符串
# 调用C++函数
res = lib.predictInit("/dev/ttyUSB0".encode('utf-8'), "project/predictor4linux/assert/nx/".encode('utf-8'), 3)
print("predictInit res: ", res)

img = "assert/img/39_NZ53M7EC4X_82_1680353113862_1680353114479.jpg".encode('utf-8')
res = lib.predict(img, 0.6)
print("predict res: ", res.decode('utf-8'))

lib.registByImg.argtypes = [c_char_p, c_int, c_char_p, c_int, c_float]
lib.registByImg.restype = c_char_p
res = lib.registByImg(img, 0, "&*YKHIhfu%^d!#@".encode('utf-8'), 0, 0.6)
print("regist res: ", res.decode('utf-8'))
