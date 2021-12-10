#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../My_CUDA.hpp"
#include <time.h>
#include <iostream>
using namespace std;

template <typename T>
__global__ void addkernel(T* in1, T* in2, T* out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    out[index] = in1[index] + in2[index];
}


template <typename T>
cudaError_t addWithCuda(T* in1_h, T* in2_h, T* out_h, unsigned int size) {
    T* in1_d = 0;
    T* in2_d = 0;
    T* out_d = 0;
    cudaError_t status;
    try {

        const unsigned blockSize = 32;
        unsigned int alignSize = ((size + blockSize - 1) / blockSize) * blockSize;
        // malloc mem in gpu

        status = cudaMalloc((void**)&in1_d, alignSize * sizeof(T));
        if (checkCudaError(status)) throw "Error";
        status = cudaMalloc((void**)&in2_d, alignSize * sizeof(T));
        if (checkCudaError(status)) throw "Error";
        status = cudaMalloc((void**)&out_d, alignSize * sizeof(T));
        if (checkCudaError(status)) throw "Error";
        status = cudaMemset(in1_d, 0, alignSize * sizeof(T));
        if (checkCudaError(status)) throw "Error";
        status = cudaMemset(in2_d, 0, alignSize * sizeof(T));
        if (checkCudaError(status)) throw "Error";
        status = cudaMemset(out_d, 0, alignSize * sizeof(T));
        if (checkCudaError(status)) throw "Error";
        status = cudaMemcpy(in1_d, in1_h, size * sizeof(T), cudaMemcpyHostToDevice);
        if (checkCudaError(status)) throw "Error";
        status = cudaMemcpy(in2_d, in2_h, size * sizeof(T), cudaMemcpyHostToDevice);
        if (checkCudaError(status)) throw "Error";

        dim3 grid((size + blockSize - 1) / blockSize);
        dim3 block(blockSize);
        double iStart = cpuSecond();
        addkernel<T><<< grid, block >>>(in1_d, in2_d, out_d);
        status = cudaDeviceSynchronize();
        if (checkCudaError(status)) throw "Error";
        double iElaps = cpuSecond() - iStart;
        printf("The time of kernel function using is %f s\n", iElaps);

        status = cudaMemcpy(out_h, out_d, size * sizeof(T), cudaMemcpyDeviceToHost);
        if (checkCudaError(status)) throw "Error";

        cudaFree(in1_d);
        cudaFree(in2_d);
        cudaFree(out_d);
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        cudaFree(in1_d);
        cudaFree(in2_d);
        cudaFree(out_d);
    }
    return status;
}


int main() {

    time_t t;
    srand((unsigned int) time(&t));

    int size = 2222222;

    // generate data
    int* in1 = static_cast<int*>(malloc(size * sizeof(int)));
    int* in2 = static_cast<int*>(malloc(size * sizeof(int)));
    int* out_gpu = static_cast<int*>(malloc(size * sizeof(int)));
    int* out_cpu = static_cast<int*>(malloc(size * sizeof(int)));
    for (int i = 0; i < size; ++i) {
        in1[i] = rand() & 0xFF;
        in2[i] = rand() & 0xFF;
        out_cpu[i] = in1[i] + in2[i];
    }

    cudaError_t status = addWithCuda<int>(in1, in2, out_gpu, size);
    CUDACHECK(status);
    // compare result
    bool result = true;
    for (int i = 0; i < size; ++i) {
        if (out_cpu[i] != out_gpu[i]) {
            result = false;
            break;
        }
    }


    // print result
    printf("Program is over\n");
    result ? printf("The result from gpu is same with cpu\n") : printf("The result from gpu isn't same with cpu\n");

    free(in1);
    free(in2);
    free(out_cpu);
    free(out_gpu);

    return 0;

}