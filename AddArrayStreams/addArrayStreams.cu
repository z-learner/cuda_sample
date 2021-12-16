#include <iostream>
#include <cstdio>
#include "cuda_runtime.h"
#include "My_CUDA.hpp"


using namespace std;

template <typename T>
void addArray(T* in1, T* in2, T* out, unsigned int size) {
    for (unsigned int i = 0; i < size; ++i)  {
        out[i] = in1[i] + in2[i];
    }
}

template <typename T>
__global__ void addkernel(T* in1, T* in2, T* out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    out[index] = in1[index] + in2[index];
}

int main() {

    setDevice(0);
    unsigned int size = 1 << 26;
    unsigned int byte = size * sizeof(float);

    unsigned int streamNumber = 8;
    unsigned int streamSize = size / streamNumber;
    unsigned int streamByte = byte / streamNumber;

    double iStart, iElaps;
    cudaStream_t streams[streamNumber];
    float* in1;
    float* in2;
    float* out;
    float* out_ref = static_cast<float*>(malloc(byte));
    CUDACHECK(cudaMallocHost((void**)&in1, byte));
    CUDACHECK(cudaMallocHost((void**)&in2, byte));
    CUDACHECK(cudaMallocHost((void**)&out, byte));
    initMem(in1, size);
    initMem(in2, size);
    // init
    initMem(in1, size);
    initMem(in2, size);

    iStart = cpuSecond();
    addArray(in1, in2, out_ref, size);
    iElaps = cpuSecond() - iStart;
    printf("cpu spend %fs in adding arrays.\n", iElaps);

    // init stream
    for (int i = 0; i < streamNumber; ++i) {
        CUDACHECK(cudaStreamCreate(&streams[i]));
    }

    dim3 block(32);
    dim3 grid((streamSize + block.x - 1) / block.x);

    iStart = cpuSecond();
    for (int i = 0; i < streamNumber; ++i) {
        unsigned int offset = i * streamSize;
        addkernel<float><<< grid, block, 0, streams[i] >>>(&in1[offset], &in2[offset], &out[offset]);
        CUDACHECK(cudaGetLastError());
    }

    for (int i = 0; i < streamNumber; ++i) {
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    iElaps = cpuSecond() - iStart;
    printf("gpu spend %fs in adding arrays.\n", iElaps);

    judgeArrayBetweenCpuAndGpuResult(out, out_ref, size) ? printf("Same.\n") : printf("Isn't same.\n");

    for (int i = 0; i < streamNumber; ++i) {
        CUDACHECK(cudaStreamDestroy(streams[i]));
    }

    CUDACHECK(cudaFreeHost(in1));
    CUDACHECK(cudaFreeHost(in2));
    CUDACHECK(cudaFreeHost(out));
    free(out_ref);
}