CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

PROJECT(Multiply)

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

FIND_PACKAGE(CUDA REQUIRED)

SET(PATH_MY_HEADER $ENV{MY_CPP_PATH})

INCLUDE_DIRECTORIES(
    ${PATH_MY_HEADER}
)

LIST(APPEND CUDA_NVCC_FLAGS -std=c++11;-O2)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME}
    multiplyMatrix.cu    
)