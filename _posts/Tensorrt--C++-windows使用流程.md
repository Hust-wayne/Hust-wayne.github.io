---
layout:     post
title:      tensorrt --- C++ -- Win 推理流程
subtitle:   yolov5--基于tensorrtx
date:       2022-01-07
author:     wayne
header-img: img/the-first.png
catalog: false
tags:
    - C++推理

---

- **tensorrt 环境配置相关**
- **C++案例流程**

***

### tensorrt环境匹配

```
1. TensorRT-7.2.3.4.Windows10.x86_64.cuda-10.2.cudnn8.1
2. vs2019
3. cmake3.20
4. tenxorrtx
```

##### 系统环境说明

Tensorrt环境配置较为简单：

- 配置tensorrt的系统环境路径
- 安装好对应的cuda、cudnn
- win 上面trt不支持python

***

#### C++推理流程

##### yolov5在win10工程编译-CMakeLists.txt

​	用Cmake工具生成工程很方便，写好cmakelists中的一些链接即可，下面是针对win的：

```cmake
cmake_minimum_required(VERSION 2.6)

project(yolov5) #1
#set(OpenCV_INCLUDE_DIRS ${OpenCV_DIR}\\include) #3
set(TRT_DIR "C:\\Program Files\\TensorRT-7.2.3.4.Windows10.x86_64.cuda-10.2.cudnn8.1\\TensorRT-7.2.3.4")  #7
set(TRT_INCLUDE_DIRS ${TRT_DIR}\\include) #8
set(TRT_LIB_DIRS ${TRT_DIR}\\lib) #9

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

# setup CUDA
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})

####
enable_language(CUDA)  # add this line, then no need to setup cuda path in vs
####
include_directories(${PROJECT_SOURCE_DIR}/include) #11
include_directories(${TRT_INCLUDE_DIRS}) #12
link_directories(${TRT_LIB_DIRS}) #13


# -D_MWAITXINTRIN_H_INCLUDED for solving error: identifier "__builtin_ia32_mwaitx" is undefined
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")

# setup opencv
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV library status:")
message(STATUS "version: ${OpenCV_VERSION}")
message(STATUS "lib path: ${OpenCV_LIB_DIRS}")
message(STATUS "Debug libraries: ${OpenCV_Debug_LIBS}")
message(STATUS "Release libraries: ${OpenCV_Release_LIBS}")
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(yolov5 ${PROJECT_SOURCE_DIR}/yolov5.cpp ${PROJECT_SOURCE_DIR}/common.hpp ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/yololayer.h)

#target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 "nvinfer" "nvinfer_plugin") #18
target_link_libraries(yolov5 cudart)
#target_link_libraries(yolov5 myplugins)
target_link_libraries(yolov5 ${OpenCV_LIBS})

#add_definitions(-O2 -pthread)
```

##### .engine文件生成及测试

使用的是pytorch版yolov5, 这里模型文件生成.wts不赘述，直接使用tensorrt-yolov5中python脚本生成即可，然后运行编译好的yolov5工程：

```
-s yolov5s.wts yolov5s.engine s  //生成engine文件
-d yolov5s.engine ./samples      //测试samples下的图像文件
```

#### 编译过程中的问题以及vs生成问题

- ##### CMake Error: No CUDA toolset found. 之前编译通过了，后重装vs2019,再次编译遇到无cuda错误，重新安装一次cuda10.2 解决问题；

- VS编译时遇到无法定位到dirent.h问题，这是由于linux与win的差异造成，在网上找到dirent.h放在Vs2019安装相关路径中解决；

  

