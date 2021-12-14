#include <iostream>
#include <cstdio>
#include <sys/time.h>
#include "cuda_runtime.h"
#include "My_CUDA.hpp"

#define RADIUS 4
#define SharedSize 32
using namespace std;

__constant__ int coef[2 * RADIUS + 1];

__global__ void stencilGlobalKernel(float* in, float* out,  int* para, unsigned int iSize, unsigned int oSize) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx + 2 * RADIUS >= iSize && idx >= oSize) return;
    for (int i = 0; i < 2 * RADIUS + 1; ++i) {
        out[idx] += para[i] * in[idx + i];
    }
}

__global__ void stencilConstantKernel(float* in, float* out, unsigned int iSize, unsigned int oSize) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx + 2 * RADIUS >= iSize && idx >= oSize) return;
    for (int i = 0; i < 2 * RADIUS + 1; ++i) {
        out[idx] += coef[i] * in[idx + i];
    }
}

__global__ void stencilSharedMemKernel(float* in, float* out, int* para, unsigned int iSize, unsigned int oSize) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sharedMem[SharedSize + 2 * RADIUS];

    if (idx + 2 * RADIUS >= iSize && idx >= oSize) return;
    unsigned int tid = threadIdx.x;
    sharedMem[tid] = in[idx];
    if (tid >= blockDim.x - 2 * RADIUS) {
        sharedMem[tid + 2 * RADIUS] = in[idx + 2 * RADIUS]; 
    }

    __syncthreads();

    for (int i = 0; i < 2 * RADIUS + 1; ++i) {
        out[idx] += para[i] * sharedMem[tid + i];
    }
}

__global__ void stencilConstantSharedMemKernel(float* in, float* out, unsigned int iSize, unsigned int oSize) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sharedMem[SharedSize + 2 * RADIUS];

    if (idx + 2 * RADIUS >= iSize && idx >= oSize) return;
    unsigned int tid = threadIdx.x;
    sharedMem[tid] = in[idx];
    if (tid >= blockDim.x - 2 * RADIUS) {
        sharedMem[tid + 2 * RADIUS] = in[idx + 2 * RADIUS]; 
    }

    __syncthreads();

    for (int i = 0; i < 2 * RADIUS + 1; ++i) {
        out[idx] += coef[i] * sharedMem[tid + i];
    }
}

cudaError_t testStencil(float* in_h, float* out_ref, int* para, unsigned int iSize, unsigned int oSize) {
    cudaError_t status = cudaSuccess;
    unsigned int iByte = iSize * sizeof(float);
    unsigned int oByte = oSize * sizeof(float);
    double iStart, iElaps;
    float* out = static_cast<float*>(malloc(oByte));
    float* in_d = 0;
    float* out_d = 0;
    int* para_d = 0;
    status = cudaMalloc((void**)&in_d, iByte);
    CUDACHECK(status);
    status = cudaMalloc((void**)&out_d, oByte);
    CUDACHECK(status);
    status = cudaMalloc((void**)&para_d, 9 * sizeof(int));
    CUDACHECK(status);

    status = cudaMemcpy(in_d, in_h, iByte, cudaMemcpyHostToDevice);
    CUDACHECK(status);
    status = cudaMemcpy(para_d, para, 9 * sizeof(int), cudaMemcpyHostToDevice);
    CUDACHECK(status);


    dim3 block(SharedSize);
    dim3 grid((oSize + SharedSize - 1) / SharedSize);
    // Global
    status = cudaMemset(out_d, 0, oByte);
    CUDACHECK(status);
    iStart = cpuSecond();
    stencilGlobalKernel<<< grid, block >>>(in_d, out_d, para_d, iSize, oSize);
    status = cudaGetLastError();
    CUDACHECK(status);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);
    iElaps = cpuSecond() - iStart;
    printf("gpu Global spend %fs in senciling.\n", iElaps);
    status = cudaMemcpy(out, out_d, oByte, cudaMemcpyDeviceToHost);
    CUDACHECK(status);
    judgeArrayBetweenCpuAndGpuResult(out_ref, out, oSize) ? printf("Same.\n") : printf("Isn't same.\n");

    // Constant
    status = cudaMemset(out_d, 0, oByte);
    CUDACHECK(status);
    iStart = cpuSecond();
    stencilConstantKernel<<< grid, block >>>(in_d, out_d, iSize, oSize);
    status = cudaGetLastError();
    CUDACHECK(status);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);
    iElaps = cpuSecond() - iStart;
    printf("gpu Constant spend %fs in senciling.\n", iElaps);
    status = cudaMemcpy(out, out_d, oByte, cudaMemcpyDeviceToHost);
    CUDACHECK(status);
    judgeArrayBetweenCpuAndGpuResult(out_ref, out, oSize) ? printf("Same.\n") : printf("Isn't same.\n");

    // Shared Memory
    status = cudaMemset(out_d, 0, oByte);
    CUDACHECK(status);
    iStart = cpuSecond();
    stencilSharedMemKernel<<< grid, block >>>(in_d, out_d, para_d, iSize, oSize);
    status = cudaGetLastError();
    CUDACHECK(status);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);
    iElaps = cpuSecond() - iStart;
    printf("gpu shared memory spend %fs in senciling.\n", iElaps);
    status = cudaMemcpy(out, out_d, oByte, cudaMemcpyDeviceToHost);
    CUDACHECK(status);
    judgeArrayBetweenCpuAndGpuResult(out_ref, out, oSize) ? printf("Same.\n") : printf("Isn't same.\n");

    // Shared Memory Constant
    status = cudaMemset(out_d, 0, oByte);
    CUDACHECK(status);
    iStart = cpuSecond();
    stencilConstantSharedMemKernel<<< grid, block >>>(in_d, out_d, iSize, oSize);
    status = cudaGetLastError();
    CUDACHECK(status);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);
    iElaps = cpuSecond() - iStart;
    printf("gpu shared memory Constant spend %fs in senciling.\n", iElaps);
    status = cudaMemcpy(out, out_d, oByte, cudaMemcpyDeviceToHost);
    CUDACHECK(status);
    judgeArrayBetweenCpuAndGpuResult(out_ref, out, oSize) ? printf("Same.\n") : printf("Isn't same.\n");


    free(out);
    cudaFree(out_d);
    cudaFree(in_d);
    cudaFree(para_d);
    return status;
}

int main() {
    setDevice(0);
    unsigned int oSize = 1 << 28;
    unsigned int iSize = oSize + 2 * RADIUS + 1;
    unsigned int iByte = iSize * sizeof(float);
    unsigned int oByte = oSize * sizeof(float);
    float* in = static_cast<float*>(malloc(iByte));
    float* out = static_cast<float*>(malloc(oByte));
    double iStart, iElaps;
    initMem(in, iSize);
    memset(out, 0, oByte);
    int para[9] = {4, 3, 2, 1, 0, 1, 2, 3, 4};
    CUDACHECK(cudaMemcpyToSymbol(coef, para, 9 * sizeof(int), 0, cudaMemcpyHostToDevice));

    // cpu
    iStart = cpuSecond();
    for (int i = 0; i < oSize; ++i) {
        for (int j = 0; j < 2 * RADIUS + 1; ++j) {
            out[i] += para[j] * in[i + j];
        }
    }
    iElaps = cpuSecond() - iStart;
    printf("cpu spend %fs in senciling.\n", iElaps);
    testStencil(in, out, para, iSize, oSize);
}