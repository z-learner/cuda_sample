#include <iostream>
#include <stdio.h>
#include "My_CUDA.hpp"
#include <sys/time.h>


__global__ void addkernel(float* in1, float* in2, float* out, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        out[index] = in1[index] + in2[index];
}


void addArray(float* in1, float* in2, float* out, unsigned int size) {
    for (int i = 0; i < size; ++i) {
        out[i] = in1[i] + in2[i];
    }
}


int main() {
    
    srand(time(NULL));
    unsigned int isize = 1 << 20;
    unsigned int nbyte = isize * sizeof(float);
    double iStart, iElaps;
    unsigned int blockSize = 512;
    // malloc data for input
    float* in1_h = static_cast<float*>(malloc(nbyte));
    float* in2_h = static_cast<float*>(malloc(nbyte));
    float* out_h = static_cast<float*>(malloc(nbyte));
    float* out_gpu = static_cast<float*>(malloc(nbyte));
    float* in1_d = 0;
    float* in2_d = 0;
    float* out_d = 0;
    cudaMalloc((void**)&in1_d, nbyte);
    cudaMalloc((void**)&in2_d, nbyte);
    cudaMalloc((void**)&out_d, nbyte);

    float* in1_host = 0;
    float* in2_host = 0;
    float* in1_host_d = 0;
    float* in2_host_d = 0;
    cudaMallocHost((void**)&in1_host, nbyte, cudaHostAllocMapped);
    cudaMallocHost((void**)&in2_host, nbyte, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&in1_host_d, (void*)in1_host, 0);
    cudaHostGetDevicePointer((void**)&in2_host_d, (void*)in2_host, 0);
    

    for (int i = 0; i < isize; ++i) {
        in1_h[i] = rand() & 0xFF;
        in2_h[i] = rand() & 0xFF;
        in1_host[i] = in1_h[i];
        in2_host[i] = in2_h[i];
    }
    cudaMemcpy(in1_d, in1_h, nbyte, cudaMemcpyHostToDevice);
    cudaMemcpy(in2_d, in2_h, nbyte, cudaMemcpyHostToDevice);


    // CPU
    iStart = cpuSecond();
    addArray(in1_h, in2_h, out_h, isize);

    iElaps = cpuSecond() - iStart;

    printf("cup spends %f s\n", iElaps);

    dim3 block(blockSize);
    dim3 grid((isize + blockSize - 1) / blockSize);
    // GPU
    iStart = cpuSecond();

    addkernel<<< grid, block >>>(in1_d, in2_d, out_d, isize);
    cudaDeviceSynchronize();

    iElaps = cpuSecond() - iStart;
    printf("gpu spends %f s\n", iElaps);
    cudaMemcpy(out_gpu, out_d, nbyte, cudaMemcpyDeviceToHost);
    judgeArrayBetweenCpuAndGpuResult(out_h, out_gpu, isize) ? printf("Right !\n") : printf("Nop !\n");


    // GPU Zero Copy
    iStart = cpuSecond();

    addkernel<<< grid, block >>>(in1_host_d, in2_host_d, out_d, isize);
    cudaDeviceSynchronize();

    iElaps = cpuSecond() - iStart;
    printf("gpu zero copy spends %f s\n", iElaps);
    cudaMemcpy(out_gpu, out_d, nbyte, cudaMemcpyDeviceToHost);
    judgeArrayBetweenCpuAndGpuResult(out_h, out_gpu, isize) ? printf("Right !\n") : printf("Nop !\n");

    cudaFreeHost(in1_host);
    cudaFreeHost(in2_host);
    cudaFree(in1_d);
    cudaFree(in2_d);
    cudaFree(out_d);
    free(in1_h);
    free(in2_h);
    free(out_h);
    free(out_gpu);
}