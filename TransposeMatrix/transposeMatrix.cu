#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "My_CUDA.hpp"
#include <sys/time.h>

using namespace std;


// CPU 
void transposeMatrix(float* in, float* out, unsigned int nx, unsigned int ny) {
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

// GPU Just copy
__global__ void matrixRowKernel(float* in, float* out, unsigned int nx, unsigned int ny) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    // if (index_x == 0 && index_y == 0) printf("in_d[0] = %f\n", in[0]);
    if (index_x < nx && index_y < ny)
        out[index_y * nx + index_x] = in[index_y * nx + index_x]; 
}

__global__ void matrixColKernel(float* in, float* out, unsigned int nx, unsigned int ny) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    // if (index_x == 0 && index_y == 0) printf("in_d[0] = %f\n", in[0]);
    if (index_x < nx && index_y < ny)
        out[index_x * ny + index_y] = in[index_x * ny + index_y]; 
}


// GPU
__global__ void transposeMatrixRowKernel(float* in, float* out, unsigned int nx, unsigned int ny) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    // if (index_x == 0 && index_y == 0) printf("in_d[0] = %f\n", in[0]);
    if (index_x < nx && index_y < ny)
        out[index_x * ny + index_y] = in[index_y * nx + index_x]; 
}

__global__ void transposeMatrixColKernel(float* in, float* out, unsigned int nx, unsigned int ny) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    // if (index_x == 0 && index_y == 0) printf("in_d[0] = %f\n", in[0]);
    if (index_x < nx && index_y < ny)
        out[index_y * nx + index_x] = in[index_x * ny + index_y]; 
}


// GPU Unroll4
__global__ void transposeMatrixRowUnroll4Kernel(float* in, float* out, unsigned int nx, unsigned int ny) {
    unsigned int index_x = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    unsigned int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    // if (index_x == 0 && index_y == 0) printf("in_d[0] = %f\n", in[0]);
    unsigned int ti = index_y * nx + index_x;
    unsigned int to = index_x * ny + index_y;
    if (index_x + 3 * blockDim.x < nx && index_y < ny) {
        out[to] = in[ti];
        out[to + ny * blockDim.x] = in[ti + blockDim.x];
        out[to + 2 * ny * blockDim.x] = in[ti + 2 * blockDim.x];
        out[to + 3 * ny * blockDim.x] = in[ti + 3 * blockDim.x];
    } 
}

__global__ void transposeMatrixColUnroll4Kernel(float* in, float* out, unsigned int nx, unsigned int ny) {
    unsigned int index_x = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    unsigned int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    // if (index_x == 0 && index_y == 0) printf("in_d[0] = %f\n", in[0]);
    unsigned int ti = index_x * ny + index_y;
    unsigned int to = index_y * nx + index_x;
    if (index_x + 3 * blockDim.x < nx && index_y < ny) {
        out[to] = in[ti];
        out[to + blockDim.x] = in[ti + nx * blockDim.x];
        out[to + 2 * blockDim.x] = in[ti + 2 * nx * blockDim.x];
        out[to + 3 * blockDim.x] = in[ti + 3 * nx * blockDim.x];
    } 
}


cudaError_t testTransport(unsigned int blockSizeX, unsigned int blockSizeY) {
    printf("blockSizeX : %d, blockSizeY : %d.\n", blockSizeX, blockSizeY);
    srand(time(NULL));
    // setDevice(0);
    
    cudaError_t status = cudaSuccess;
    unsigned int ix = 1 << 13;
    unsigned int iy = 1 << 13;
    unsigned int iSize = ix * iy;
    unsigned int iBytes = iSize * sizeof(float);
    double iStart, iElaps;
    float* in = static_cast<float*>(malloc(iBytes));
    float* out = static_cast<float*>(malloc(iBytes));
    float* out_refgpu = static_cast<float*>(malloc(iBytes));
    // memset(out_refgpu, 0xFF, iBytes);

    for (int i = 0; i < iSize; ++i) {
        in[i] = (rand() & 0xFF) / 10.0;
        out[i] = in[i];
    }

    float* in_d = 0;
    float* out_d = 0;
    status = cudaMalloc((void**)&in_d, iBytes);
    CUDACHECK(status);
    status = cudaMalloc((void**)&out_d, iBytes);
    CUDACHECK(status);
    status = cudaMemcpy(in_d, in, iBytes, cudaMemcpyHostToDevice);
    CUDACHECK(status);
    

    dim3 block(blockSizeX, blockSizeY);
    dim3 grid((ix + blockSizeX - 1) / blockSizeX, (iy + blockSizeY - 1) / blockSizeY);
    
    printf("block.x : %d, block.y : %d.\n", block.x, block.y);
    printf("grid.x : %d, grid.y : %d.\n", grid.x, grid.y);
    
    // myPrintfFloatGpu(out_d, 10);
    // myPrintfFloat(out_refgpu, 10);

    /* just make gpu ready to run real program */ 
    matrixRowKernel<<< grid, block >>>(in_d, out_d, ix, iy);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);
    matrixColKernel<<< grid, block >>>(in_d, out_d, ix, iy);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);

    
    // GPU Copy
    status = cudaMemset(out_d, 0x00, iBytes);
    CUDACHECK(status);
    iStart = cpuSecond();
    matrixRowKernel<<<grid, block>>>(in_d, out_d, ix, iy);
    status = cudaGetLastError();
    CUDACHECK(status);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);
    iElaps = cpuSecond() - iStart;
    printf("GPU spends %f to copy matrix by row.\n", iElaps);
    status = cudaMemcpy(out_refgpu, out_d, iBytes, cudaMemcpyDeviceToHost);
    CUDACHECK(status);
    judgeArrayBetweenCpuAndGpuResult(out, out_refgpu, iSize) ? printf("Same.\n") : printf("Isn't same.\n");
    // myPrintfFloat(in, 10);
    // myPrintfFloatGpu(in_d, 10);
    // myPrintfFloat(out, 10);
    // myPrintfFloatGpu(out_d, 10);
    // myPrintfFloat(out_refgpu, 10);

    status = cudaMemset(out_d, 0x00, iBytes);
    CUDACHECK(status);
    iStart = cpuSecond();
    matrixColKernel<<< grid, block >>>(in_d, out_d, ix, iy);
    status = cudaGetLastError();
    CUDACHECK(status);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);
    iElaps = cpuSecond() - iStart;
    printf("GPU spends %f to copy matrix by col.\n", iElaps);
    status = cudaMemcpy(out_refgpu, out_d, iBytes, cudaMemcpyDeviceToHost);
    CUDACHECK(status);
    judgeArrayBetweenCpuAndGpuResult(out, out_refgpu, iSize) ? printf("Same.\n") : printf("Isn't same.\n");



    transposeMatrix(in, out, ix, iy);

    // GPU Transpose 
    status = cudaMemset(out_d, 0x00, iBytes);
    CUDACHECK(status);
    iStart = cpuSecond();
    transposeMatrixRowKernel<<< grid, block >>>(in_d, out_d, ix, iy);
    status = cudaGetLastError();
    CUDACHECK(status);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);
    iElaps = cpuSecond() - iStart;
    printf("GPU spends %f to transpose matrix by row.\n", iElaps);
    status = cudaMemcpy(out_refgpu, out_d, iBytes, cudaMemcpyDeviceToHost);
    CUDACHECK(status);
    judgeArrayBetweenCpuAndGpuResult(out, out_refgpu, iSize) ? printf("Same.\n") : printf("Isn't same.\n");


    status = cudaMemset(out_d, 0x00, iBytes);
    CUDACHECK(status);
    iStart = cpuSecond();
    transposeMatrixColKernel<<< grid, block >>>(in_d, out_d, ix, iy);
    status = cudaGetLastError();
    CUDACHECK(status);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);
    iElaps = cpuSecond() - iStart;
    printf("GPU spends %f to transpose matrix by col.\n", iElaps);
    status = cudaMemcpy(out_refgpu, out_d, iBytes, cudaMemcpyDeviceToHost);
    CUDACHECK(status);
    judgeArrayBetweenCpuAndGpuResult(out, out_refgpu, iSize) ? printf("Same.\n") : printf("Isn't same.\n");


    // GPU Transpose Unroll
    status = cudaMemset(out_d, 0x00, iBytes);
    CUDACHECK(status);
    iStart = cpuSecond();
    transposeMatrixRowUnroll4Kernel<<< grid, block >>>(in_d, out_d, ix, iy);
    status = cudaGetLastError();
    CUDACHECK(status);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);
    iElaps = cpuSecond() - iStart;
    printf("GPU spends %f to transpose Unroll4 matrix by row.\n", iElaps);
    status = cudaMemcpy(out_refgpu, out_d, iBytes, cudaMemcpyDeviceToHost);
    CUDACHECK(status);
    judgeArrayBetweenCpuAndGpuResult(out, out_refgpu, iSize) ? printf("Same.\n") : printf("Isn't same.\n");


    status = cudaMemset(out_d, 0x00, iBytes);
    CUDACHECK(status);
    iStart = cpuSecond();
    transposeMatrixColUnroll4Kernel<<< grid, block >>>(in_d, out_d, ix, iy);
    status = cudaGetLastError();
    CUDACHECK(status);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);
    iElaps = cpuSecond() - iStart;
    printf("GPU spends %f to transpose Unroll4 matrix by col.\n", iElaps);
    status = cudaMemcpy(out_refgpu, out_d, iBytes, cudaMemcpyDeviceToHost);
    CUDACHECK(status);
    judgeArrayBetweenCpuAndGpuResult(out, out_refgpu, iSize) ? printf("Same.\n") : printf("Isn't same.\n");






    cudaFree(in_d);
    cudaFree(out_d);
    free(in);
    free(out);
    free(out_refgpu);
    return status;
}


int main(int argc, char** argv) {
    simpleDeviceQuery(0);
    
    unsigned int blockSizeX = 32;
    unsigned int blockSizeY = 32;
    if (argc > 1) blockSizeX = atoi(argv[1]);
    if (argc > 2) blockSizeY = atoi(argv[2]);
    
    CUDACHECK(testTransport(blockSizeX, blockSizeY));

}