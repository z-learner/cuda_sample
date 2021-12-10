# Can't Run
# Question : can't link -lcudadevrt
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(ReduceArray)

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

FIND_PACKAGE(CUDA REQUIRED)

SET(PATH_MY_HEADER $ENV{MY_CPP_PATH})

INCLUDE_DIRECTORIES(
    /usr/local/cuda-11.5/include
    ${PATH_MY_HEADER}
)

LINK_LIBRARIES("/usr/local/cuda-11.5/targets/x86_64-linux/lib/libcudadevrt.a")

LIST(APPEND CUDA_NVCC_FLAGS -rdc=true;-std=c++11;-O2)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME}
    reduceArray.cu    
)

# SET(CUDA_DEVRT "-lcudadevrt")
# TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${CUDA_DEVRT})
# nvcc -rdc=true reduceArray.cu -o reduceArray -lcudadevrt

MESSAGE(STATUS "**************************************")