#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "My_CUDA.hpp"

using namespace std;

template <typename T>
__global__ void addkernel(T* in1, T* in2, T* out, unsigned int nx, unsigned int ny) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    int index = nx * iy + ix;
    // int index = ny * ix + iy;
    // if (index >= nx * ny) printf("index is out of range ! \n");
    if (ix < nx && iy < ny)
        out[index] = in1[index] + in2[index];

}


template <typename T>
cudaError_t addMatrixWithCuda(T* in1_h, T* in2_h, T* out_h, unsigned int nx, unsigned int ny) {
    T* in1_d = 0;
    T* in2_d = 0;
    T* out_d = 0;
    int size = nx * ny;
    cudaError_t status;
    double iStart, iElaps;

    // malloc memory from gpu
    status = cudaMalloc((void**)&in1_d, size * sizeof(T));
    CUDACHECK(status);
    status = cudaMalloc((void**)&in2_d, size * sizeof(T));
    CUDACHECK(status);
    status = cudaMalloc((void**)&out_d, size * sizeof(T));
    CUDACHECK(status);

    // cpoy data to gpu from cpu
    status = cudaMemcpy(in1_d, in1_h, size * sizeof(T), cudaMemcpyHostToDevice);
    CUDACHECK(status);
    status = cudaMemcpy(in2_d, in2_h, size * sizeof(T), cudaMemcpyHostToDevice);
    CUDACHECK(status);

    // set configuration of kernel and start it
    dim3 block(16, 32);
    dim3 grid( (nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y );
    iStart = cpuSecond();

    addkernel<T><<< grid, block >>>(in1_d, in2_d, out_d, nx, ny);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);

    iElaps = cpuSecond() - iStart;
    printf("The time of GPU consume is %f s\n", iElaps);

    status = cudaMemcpy(out_h, out_d, size * sizeof(T), cudaMemcpyDeviceToHost);
    CUDACHECK(status);

    cudaFree(in1_d);
    cudaFree(in2_d);
    cudaFree(out_d);



    return status;

}


int main(int argc, char* argv[]) {

    printf("%s is starting ... \n", argv[0]);

    srand(time(NULL));
    double iStart, iElaps;
    int nx = 1 << 14;
    int ny = 1 << 14;
    int size = nx * ny;
    printf("Matrix is %dX%d, total has %d number\n", nx, ny, size);
    int* in1_h = static_cast<int*>(malloc(size * sizeof(int)));
    int* in2_h = static_cast<int*>(malloc(size * sizeof(int)));
    int* out_cpu = static_cast<int*>(malloc(size * sizeof(int)));
    int* out_gpu = static_cast<int*>(malloc(size * sizeof(int)));

    // generate data
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            int index = i * ny + j;
            in1_h[index] = rand() & 0XFF;
            in2_h[index] = rand() & 0XFF;
        }
    }

    // calculate result in cpu
    iStart = cpuSecond();
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            int index = i * ny + j;
            out_cpu[index] = in1_h[index] + in2_h[index];
        }
    }

    // calculate result in gpu
    setDevice(0);
    iElaps = cpuSecond() - iStart;
    printf("The time of CPU consume is %f s\n", iElaps);

    addMatrixWithCuda<int>(in1_h, in2_h, out_gpu, nx, ny);

    judgeMatrixBetweenCpuAndGpuResult(out_cpu, out_gpu, nx, ny) ? printf("The result of GPU is same with CPU's\n") : printf("The result of GPU isn't same with CPU's\n");

    free(in1_h);
    free(in2_h);
    free(out_cpu);
    free(out_gpu);
    return 0;

}