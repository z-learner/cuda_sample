
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

PROJECT(TestBranchEfficiency)

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

FIND_PACKAGE(CUDA REQUIRED)

if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)
# 打印相关信息, CMAKE_CURRENT_SOURCE_DIR指的是当前处理的CMakeLists.txt所在的路径
MESSAGE(STATUS "current path: ${CMAKE_CURRENT_SOURCE_DIR}")

SET(PATH_MY_HEADER $ENV{MY_CPP_PATH})

# 指定CUDA头文件的搜索路径
INCLUDE_DIRECTORIES(/usr/local/cuda-11.5/include)

# 头文件路径
INCLUDE_DIRECTORIES(
    ${PATH_MY_HEADER}
)

# 使CUDA NVCC 编译器支持C++11特性
LIST(APPEND CUDA_NVCC_FLAGS -std=c++11;-O2)

# close brance prediction
LIST(APPEND CUDA_NVCC_FLAGS -g;-G;-Xcompiler;-fPIC)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} 
    branchEfficiency.cu
)

MESSAGE(STATUS "**************************************")
