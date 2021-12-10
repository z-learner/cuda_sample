#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "My_CUDA.hpp"
#define warpSize 32
using namespace std;

__global__ void kernel1() {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float a, b;
    if (tid % 2 == 0 ) {
        a = 1.0f;
        b = 1.0f;
    } else {
        a = 2.0f;
        b = 2.0f;
    }
}

__global__ void kernel2() {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float a, b;
    if ((tid / warpSize) % 2 == 0 ) {
        a = 1.0f;
        b = 1.0f;
    } else {
        a = 2.0f;
        b = 2.0f;
    }
}

__global__ void kernel3() {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float a, b;
    bool ipred = (tid % 2 == 0);
    if (ipred) {
        a = 1.0f;
        b = 1.0f;
    } 
    if (!ipred) {
        a = 2.0f;
        b = 2.0f;
    }
}


cudaError_t testBranchEfficiency() {

    setDevice(0);

    dim3 block(32);
    dim3 grid(1);

    kernel1<<<grid, block>>>();
    cudaDeviceSynchronize();
    return cudaDeviceReset();

}


int main(int argc, char* argv[]) {

    CUDACHECK(testBranchEfficiency());

    return 0;

}