#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include "cuda_runtime.h"
#include "My_CUDA.hpp"

#define TILE_SIZE 32

using namespace std;


void multiplyMatrix(float* in1, float* in2, float* out, unsigned int nx1, unsigned int ny1, unsigned int nx2, unsigned int ny2) {
    unsigned int outSizeX = nx1;
    unsigned int outSizeY = ny2;

    for (unsigned int ix = 0; ix < outSizeX; ++ix) {
        for (unsigned int iy = 0; iy < outSizeY; ++iy) {
            float res = 0.0;;
            for (int i = 0; i < ny1; i++) {
                res += in1[ix * ny1 + i] * in2[i * ny2 + iy];
            }
            out[ix * outSizeY + iy] = res;
        }
    }

}

// ny2 == nx1
__global__ void multiplyMatrixKernel(float* in1, float* in2, float* out, unsigned int nx1, unsigned int ny1, unsigned int nx2, unsigned int ny2) {
    float res = 0.0f;
    // unsigned int outSizeX = nx1;
    unsigned int outSizeY = ny2;

    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    for (int i = 0; i < ny1; i++) {
        res += in1[ix * ny1 + i] * in2[i * ny2 + iy];
    }

    out[ix * outSizeY + iy] = res;

}

cudaError_t multiplyMatrixKernelWithCuda(float* in1_h, float* in2_h, float* out_h, unsigned int nx1, unsigned int ny1, unsigned int nx2, unsigned int ny2, dim3 blockSize) {
    // setDevice(0);
    cudaError_t status = cudaSuccess;
    float* in1_d;
    unsigned int iSize1 = nx1 * ny1;
    unsigned int iByte1 = iSize1 * sizeof(float);
    float* in2_d;
    unsigned int iSize2 = nx2 * ny2;
    unsigned int iByte2 = iSize2 * sizeof(float);
    float* out_d;    
    unsigned int oSize = nx1 * ny2;
    unsigned int oByte = oSize * sizeof(float);
    double iStart, iElaps;
    status = cudaMalloc((void**)&in1_d, iByte1);
    CUDACHECK(status);
    status = cudaMalloc((void**)&in2_d, iByte2);
    CUDACHECK(status);
    status = cudaMalloc((void**)&out_d, oByte);
    CUDACHECK(status);

    status = cudaMemcpy(in1_d, in1_h, iByte1, cudaMemcpyHostToDevice);
    CUDACHECK(status);
    status = cudaMemcpy(in2_d, in2_h, iByte2, cudaMemcpyHostToDevice);
    CUDACHECK(status);

    dim3 gridSize((nx1 + blockSize.x - 1) / blockSize.x, (ny2 + blockSize.y - 1) / blockSize.y);
    iStart = cpuSecond();
    multiplyMatrixKernel<<< gridSize, blockSize >>>(in1_d, in2_d, out_d, nx1, ny1, nx2, ny2);
    status = cudaGetLastError();
    CUDACHECK(status);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);
    iElaps = cpuSecond() - iStart;
    printf("gpu spend %fs in kernel function of multiplying matrix.\n", iElaps);

    status = cudaGetLastError();
    CUDACHECK(status);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);

    status = cudaMemcpy(out_h, out_d, oByte, cudaMemcpyDeviceToHost);
    CUDACHECK(status);



    cudaFree(in1_d);
    cudaFree(in2_d);
    cudaFree(out_d);

    return status;
}


__global__ void multiplyMatrixShareMemKernel(float* in1, float* in2, float* out, unsigned int nx1, unsigned int ny1, unsigned int nx2, unsigned int ny2) {
    // unsigned int outSizeX = nx1;
    unsigned int outSizeY = ny2;
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int tidx = threadIdx.x;
    unsigned int tidy = threadIdx.y;

    __shared__ float sharedMem1[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedMem2[TILE_SIZE][TILE_SIZE];

    float res = 0.0;
    for (int i = 0; i < ny1 / TILE_SIZE; ++i) {

        sharedMem1[tidx][tidy] = in1[ix * ny1 + i * TILE_SIZE + threadIdx.y];
        sharedMem2[tidx][tidy] = in2[(i * TILE_SIZE + threadIdx.x) * ny2 + iy];

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j) {
            res += sharedMem1[tidx][j] * sharedMem2[j][tidy];
        } 
        __syncthreads();

        out[ix * outSizeY + iy] = res;
    }
    
}


cudaError_t multiplyMatrixKernelSharedMemWithCuda(float* in1_h, float* in2_h, float* out_h, unsigned int nx1, unsigned int ny1, unsigned int nx2, unsigned int ny2) {
    // setDevice(0);
    cudaError_t status = cudaSuccess;
    float* in1_d;
    unsigned int iSize1 = nx1 * ny1;
    unsigned int iByte1 = iSize1 * sizeof(float);
    float* in2_d;
    unsigned int iSize2 = nx2 * ny2;
    unsigned int iByte2 = iSize2 * sizeof(float);
    float* out_d;    
    unsigned int oSize = nx1 * ny2;
    unsigned int oByte = oSize * sizeof(float);
    double iStart, iElaps;
    status = cudaMalloc((void**)&in1_d, iByte1);
    CUDACHECK(status);
    status = cudaMalloc((void**)&in2_d, iByte2);
    CUDACHECK(status);
    status = cudaMalloc((void**)&out_d, oByte);
    CUDACHECK(status);

    status = cudaMemcpy(in1_d, in1_h, iByte1, cudaMemcpyHostToDevice);
    CUDACHECK(status);
    status = cudaMemcpy(in2_d, in2_h, iByte2, cudaMemcpyHostToDevice);
    CUDACHECK(status);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((nx1 + blockSize.x - 1) / blockSize.x, (ny2 + blockSize.y - 1) / blockSize.y);
    iStart = cpuSecond();
    multiplyMatrixShareMemKernel<<< gridSize, blockSize >>>(in1_d, in2_d, out_d, nx1, ny1, nx2, ny2);
    status = cudaGetLastError();
    CUDACHECK(status);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);
    iElaps = cpuSecond() - iStart;
    printf("gpu spend sharedMem %fs in kernel function of multiplying matrix.\n", iElaps);

    status = cudaGetLastError();
    CUDACHECK(status);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);

    status = cudaMemcpy(out_h, out_d, oByte, cudaMemcpyDeviceToHost);
    CUDACHECK(status);



    cudaFree(in1_d);
    cudaFree(in2_d);
    cudaFree(out_d);

    return status;
}




int main(int argc, char** argv) {

    setDevice(0);

    unsigned int nx1 = 1 << 10;
    unsigned int ny1 = 1 << 12;
    unsigned int nx2 = 1 << 12; 
    unsigned int ny2 = 1 << 10;
    unsigned int iSize1 = nx1 * ny1;
    unsigned int iByte1 = iSize1 * sizeof(float);
    unsigned int iSize2 = nx2 * ny2;
    unsigned int iByte2 = iSize2 * sizeof(float);
    unsigned int oSize = nx1 * ny2;
    unsigned int oByte = oSize * sizeof(float);

    double iStart, iElaps;


    float* in1 = static_cast<float*>(malloc(iByte1)); 
    float* in2 = static_cast<float*>(malloc(iByte2)); 
    float* out_cpu = static_cast<float*>(malloc(oByte)); 
    float* out_gpu = static_cast<float*>(malloc(oByte)); 
    initMem(in1, iSize1);
    initMem(in2, iSize2);

    dim3 blockSize(32, 32);

    if (argc > 1) blockSize.x = atoi(argv[1]);
    if (argc > 2) blockSize.y = atoi(argv[2]);

    iStart = cpuSecond();
    multiplyMatrix(in1, in2, out_cpu, nx1, ny1, nx2, ny2);
    iElaps = cpuSecond() - iStart;
    printf("cpu spend %f s in multiplying matrix.\n", iElaps);
    
    iStart = cpuSecond();
    multiplyMatrixKernelWithCuda(in1, in2, out_gpu, nx1, ny1, nx2, ny2, blockSize);
    iElaps = cpuSecond() - iStart;
    printf("gpu totally spend %fs in multiplying matrix.\n", iElaps);
    // myPrintfFloat(out_gpu, 8);
    judgeArrayBetweenCpuAndGpuResult(out_cpu, out_gpu, oSize) ? printf("Same.\n") : printf("Isn't Same.\n");

    memset(out_gpu, 0, oByte);
    iStart = cpuSecond();
    multiplyMatrixKernelSharedMemWithCuda(in1, in2, out_gpu, nx1, ny1, nx2, ny2);
    iElaps = cpuSecond() - iStart;
    printf("gpu spend sharedMem totally spend %fs in multiplying matrix.\n", iElaps);
    // myPrintfFloat(out_gpu, 8);
    judgeArrayBetweenCpuAndGpuResult(out_cpu, out_gpu, oSize) ? printf("Same.\n") : printf("Isn't Same.\n");

    free(in1);
    free(in2);
    free(out_cpu);
    free(out_gpu);

}