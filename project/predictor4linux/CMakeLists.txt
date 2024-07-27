cmake_minimum_required(VERSION 3.11)
project(smart_predictor)
 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_COMPILER "g++")

add_library(
    smart_predictor 
    SHARED 
        smart_predictor.cpp
        classifierFactory.cpp
        classifierFactory.h
        classifierCamera.cpp
        classifierCamera.h
        IClassifier.h
        encryptor.cpp
        encryptor.h
        index.cpp
        index.h
        utils.h
)
# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(spdlog REQUIRED)

# 添加头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OPENSSL_INCLUDE_DIR})
# 添加链接目录
set(camera_DIR ${CMAKE_SOURCE_DIR}/lib/camera)
find_library(Camera_LIBRARY NAMES RvPredC PATHS ${camera_DIR} NO_DEFAULT_PATH)

# 链接opencv库，your_executable_or_library表示可执行文件名
target_link_libraries(
    smart_predictor
    ${OPENSSL_LIBRARIES}
    ${OpenCV_LIBS}
    ${Camera_LIBRARY}
)

