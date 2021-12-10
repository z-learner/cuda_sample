#CMAKE FOR CUDA

# SET version
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

# set project name
PROJECT(SumMatrix)

# Set the output folder where your program will be created
set (EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

#// cmake寻找cuda，并且CUDA_ADD_EXECUTABLE等cuda前缀的命令是由cuda提供的，cmake本身不支持
FIND_PACKAGE(CUDA REQUIRED)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

# 打印相关信息, CMAKE_CURRENT_SOURCE_DIR指的是当前处理的CMakeLists.txt所在的路径
MESSAGE(STATUS "current path: ${CMAKE_CURRENT_SOURCE_DIR}")

SET(PATH_MY_HEADER $ENV{MY_CPP_PATH})
SET(CPP_FILE .)
SET(CU_FILE .)

#  递归查询所有匹配的文件:*.cpp和*.cu
FILE(GLOB_RECURSE CPP_LIST ${CPP_FILE}/*.cpp)
FILE(GLOB_RECURSE CU_LIST ${CU_FILE}/*.cu)
MESSAGE(STATUS "cpp list: ${CPP_LIST}")
MESSAGE(STATUS "cu list: ${CU_LIST}")

# 使CMake支持C++11特性
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu++0x")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")

# 指定CUDA头文件的搜索路径
INCLUDE_DIRECTORIES(/usr/local/cuda-11.5/include)
# 头文件路径
INCLUDE_DIRECTORIES(
    ${PATH_MY_HEADER}
)
# 使CUDA NVCC 编译器支持C++11特性
LIST(APPEND CUDA_NVCC_FLAGS -std=c++11;-O2)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

# 编译可执行程序，即生成可执行文件ProJECT
#ADD_EXECUTABLE(CUDA_Test ${CPP_LIST} ${CU_LIST}) # 不包括编译*.cu文件，需要使用CUDA_ADD_EXECUTABLE命令
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${CU_LIST})
# 链接外部库，需要包含cudalib like cuDNN
# TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${CUDA_cudadevrt_LIBRARY}
MESSAGE(STATUS "**************************************")
