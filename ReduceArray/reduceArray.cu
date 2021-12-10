#include "../My_CUDA.hpp"
#include <stdio.h>
#include <iostream>
#include <iostream>
#include <time.h>

#define blockSize 512

using namespace std;




/* Unwarp */
template <typename T>
__global__ void reduceKernelUnwarp(T* in, T* out, unsigned int alignSize) {
    // locate data
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    T* idata = in + blockDim.x * blockIdx.x;

    if (idx >= alignSize) return;

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) idata[tid] += idata[tid + stride];

        __syncthreads();
    } 

    if (tid < 32) {
        volatile T* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }


    if (tid == 0) out[blockIdx.x] = idata[0];
}   
// int blockSize = 64; must = 64 * n 
template <typename T>
cudaError_t reduceKernelWithCuda(T* in_h, T* out, unsigned int size) {
    // int blockSize = 64;
    double iStart, iElaps;
    cudaError_t status;
    T* in_d;
    T* out_d;
    unsigned int alignSize = (size + blockSize - 1) / blockSize * blockSize;
    status = cudaMalloc((void**)&in_d, alignSize * sizeof(T));
    CUDACHECK(status);
    status = cudaMemset(in_d, 0, alignSize * sizeof(T));
    CUDACHECK(status);
    status = cudaMemcpy(in_d, in_h, size * sizeof(T), cudaMemcpyHostToDevice);
    CUDACHECK(status);
    status = cudaMalloc((void**)&out_d, alignSize * sizeof(T) / blockSize);
    CUDACHECK(status);
    
    T* out_h = static_cast<T*>(malloc(alignSize / blockSize * sizeof(T)));

    dim3 block(blockSize);
    dim3 grid(alignSize / blockSize);

    iStart = cpuSecond();
    reduceKernelUnwarp<T><<< grid, block >>>(in_d, out_d, alignSize);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);

    status = cudaMemcpy(out_h, out_d, alignSize / blockSize * sizeof(T), cudaMemcpyDeviceToHost);
    CUDACHECK(status);

    // printf("alignSize : %d\n", alignSize);
    // printf("output array has %d nnumbers\n", alignSize / blockSize);
    for (int i = 0; i < alignSize / blockSize; ++i) {
        // printf("%d ", out_h[i]);
        *out += out_h[i];
    }
    // printf("\n");

    iElaps = cpuSecond() - iStart;
    printf("GPU Unwarp spend %f s in calculating\n", iElaps);

    return status;

}


/* Uwarp Unroll8 */
template <typename T>
__global__ void reduceKernelUnwarpUnroll8(T* in, T* out, unsigned int alignSize) {
    // locate data
    int idx = 8 * blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    T* idata = in + 8 * blockDim.x * blockIdx.x;
    // if (idx >= alignSize) return;

    // if (idx + 7 * blockDim.x < alignSize) {
        T a1 = in[idx];
        T a2 = in[idx + blockDim.x];
        T a3 = in[idx + 2 * blockDim.x];
        T a4 = in[idx + 3 * blockDim.x];
        T a5 = in[idx + 4 * blockDim.x];
        T a6 = in[idx + 5 * blockDim.x];
        T a7 = in[idx + 6 * blockDim.x];
        T a8 = in[idx + 7 * blockDim.x];
    // }

    in[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) idata[tid] += idata[tid + stride];

        __syncthreads();
    } 

    if (tid < 32) {
        volatile T* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }


    if (tid == 0) out[blockIdx.x] = idata[0];
} 
// grip / 8 = m
// int blockSize = 64; must = 64 * n 
template <typename T>
cudaError_t reduceKernelUnwarpUroll8WithCuda(T* in_h, T* out, unsigned int size) {
    // int blockSize = 64;
    double iStart, iElaps;
    cudaError_t status;
    T* in_d;
    T* out_d;
    unsigned int alignSize = (size + blockSize - 1) / blockSize * blockSize;
    status = cudaMalloc((void**)&in_d, alignSize * sizeof(T));
    CUDACHECK(status);
    status = cudaMemset(in_d, 0, alignSize * sizeof(T));
    CUDACHECK(status);
    status = cudaMemcpy(in_d, in_h, size * sizeof(T), cudaMemcpyHostToDevice);
    CUDACHECK(status);
    status = cudaMalloc((void**)&out_d, alignSize * sizeof(T) / blockSize);
    CUDACHECK(status);
    status = cudaMemset(out_d, 0, alignSize / blockSize * sizeof(T));
    CUDACHECK(status);  

    T* out_h = static_cast<T*>(malloc(alignSize / blockSize * sizeof(T)));

    dim3 block(blockSize);
    dim3 grid((alignSize / blockSize) / 8);

    iStart = cpuSecond();
    reduceKernelUnwarpUnroll8<T><<< grid, block >>>(in_d, out_d, alignSize);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);

    status = cudaMemcpy(out_h, out_d, alignSize / blockSize * sizeof(T), cudaMemcpyDeviceToHost);
    CUDACHECK(status);

    // printf("alignSize : %d\n", alignSize);
    // printf("output array has %d nnumbers\n", alignSize / blockSize);
    for (int i = 0; i < alignSize / blockSize; ++i) {
        // printf("%d ", out_h[i]);
        *out += out_h[i];
    }
    // printf("\n");

    iElaps = cpuSecond() - iStart;
    printf("GPU Unwarp Unroll8 spend %f s in calculating\n", iElaps);

    return status;

}



/*UnWarp Unroll8 Complete*/
// blockSize <= 1024
template <typename T>
__global__ void reduceKernelUnwarpUnroll8Complete(T* in, T* out, unsigned int alignSize) {
    // locate data
    int idx = 8 * blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    T* idata = in + 8 * blockDim.x * blockIdx.x;
    // if (idx >= alignSize) return;

    // if (idx + 7 * blockDim.x < alignSize) {
        T a1 = in[idx];
        T a2 = in[idx + blockDim.x];
        T a3 = in[idx + 2 * blockDim.x];
        T a4 = in[idx + 3 * blockDim.x];
        T a5 = in[idx + 4 * blockDim.x];
        T a6 = in[idx + 5 * blockDim.x];
        T a7 = in[idx + 6 * blockDim.x];
        T a8 = in[idx + 7 * blockDim.x];
    // }

    in[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];
	__syncthreads();


    if (tid < 32) {
        volatile T* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }


    if (tid == 0) out[blockIdx.x] = idata[0];
} 

// grip / 8 = m
// int blockSize = 64; must = 64 * n 
template <typename T>
cudaError_t reduceKernelUnwarpUroll8WithCudaComplete(T* in_h, T* out, unsigned int size) {
    // int blockSize = 64;
    double iStart, iElaps;
    cudaError_t status;
    T* in_d;
    T* out_d;
    unsigned int alignSize = (size + blockSize - 1) / blockSize * blockSize;
    status = cudaMalloc((void**)&in_d, alignSize * sizeof(T));
    CUDACHECK(status);
    status = cudaMemset(in_d, 0, alignSize * sizeof(T));
    CUDACHECK(status);
    status = cudaMemcpy(in_d, in_h, size * sizeof(T), cudaMemcpyHostToDevice);
    CUDACHECK(status);
    status = cudaMalloc((void**)&out_d, alignSize * sizeof(T) / blockSize);
    CUDACHECK(status);
    status = cudaMemset(out_d, 0, alignSize / blockSize * sizeof(T));
    CUDACHECK(status);  

    T* out_h = static_cast<T*>(malloc(alignSize / blockSize * sizeof(T)));

    dim3 block(blockSize);
    dim3 grid((alignSize / blockSize) / 8);

    iStart = cpuSecond();
    reduceKernelUnwarpUnroll8Complete<T><<< grid, block >>>(in_d, out_d, alignSize);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);

    status = cudaMemcpy(out_h, out_d, alignSize / blockSize * sizeof(T), cudaMemcpyDeviceToHost);
    CUDACHECK(status);

    // printf("alignSize : %d\n", alignSize);
    // printf("output array has %d nnumbers\n", alignSize / blockSize);
    for (int i = 0; i < alignSize / blockSize; ++i) {
        // printf("%d ", out_h[i]);
        *out += out_h[i];
    }
    // printf("\n");

    iElaps = cpuSecond() - iStart;
    printf("GPU Unwarp Unroll8 spend %f s in calculating\n", iElaps);

    return status;

}


/* Recursive */
template <typename T>
__global__ void reduceRecuriveKernel(T* g_idata, T* g_odata, unsigned int size) {

    unsigned int tid = threadIdx.x;
    T* idata = g_idata + blockDim.x * blockIdx.x;
    T* odata = &g_odata[blockIdx.x];

    // stop condition
    if (size == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    // nested invocation
    int iStride = size >> 1;
    if (iStride > 1 && tid < iStride) {
        idata[tid] += idata[tid + iStride];
    }

    __syncthreads();

    // nested invocation to generate child gride
    if (tid == 0) {
        reduceRecuriveKernel<T><<< 1, iStride >>>(idata, odata, iStride);

        cudaDeviceSynchronize();
    }

    __syncthreads();

}
template <typename T>
cudaError_t reduceRecuriveKernelWithCuda(T* in_h, T* out, unsigned int size) {
    // int blockSize = 64;
    double iStart, iElaps;
    cudaError_t status;
    T* in_d;
    T* out_d;
    unsigned int alignSize = (size + blockSize - 1) / blockSize * blockSize;
    status = cudaMalloc((void**)&in_d, alignSize * sizeof(T));
    CUDACHECK(status);
    status = cudaMemset(in_d, 0, alignSize * sizeof(T));
    CUDACHECK(status);
    status = cudaMemcpy(in_d, in_h, size * sizeof(T), cudaMemcpyHostToDevice);
    CUDACHECK(status);
    status = cudaMalloc((void**)&out_d, alignSize * sizeof(T) / blockSize);
    CUDACHECK(status);
    status = cudaMemset(out_d, 0, alignSize / blockSize * sizeof(T));
    CUDACHECK(status);  

    T* out_h = static_cast<T*>(malloc(alignSize / blockSize * sizeof(T)));

    dim3 block(blockSize);
    dim3 grid(alignSize / blockSize);

    iStart = cpuSecond();
    // ----------------------------------------------------- size = block.x
    reduceRecuriveKernel<T><<< grid, block >>>(in_d, out_d, block.x);
    status = cudaDeviceSynchronize();
    CUDACHECK(status);

    status = cudaMemcpy(out_h, out_d, alignSize / blockSize * sizeof(T), cudaMemcpyDeviceToHost);
    CUDACHECK(status);

    // printf("alignSize : %d\n", alignSize);
    // printf("output array has %d nnumbers\n", alignSize / blockSize);
    for (int i = 0; i < alignSize / blockSize; ++i) {
        // printf("%d ", out_h[i]);
        *out += out_h[i];
    }
    // printf("\n");

    iElaps = cpuSecond() - iStart;
    printf("GPU Recurive spend %f s in calculating\n", iElaps);

    return status;

}



int main(int argc, char* argv[]) {

    srand(time(NULL));
    double iStart, iElaps;
    unsigned int size = 1 << 24;

    int result_cpu = 0, result_gpu = 0;

    int* in = static_cast<int*>(malloc(size * sizeof(int)));
    for (int i = 0; i < size; ++i) {
        in[i] = rand() & 0xFF;
        // printf("%d ", in[i]);
    }
    // printf("\n");
    iStart = cpuSecond();

    for (int i = 0; i < size; ++i) result_cpu += in[i];

    iElaps = cpuSecond() - iStart;
    printf("CPU spend %f s in calculating\n", iElaps);

    CUDACHECK(reduceKernelWithCuda<int>(in, &result_gpu, size));

    // printf("the result of cpu is %d\n", result_cpu);
    // printf("the result of gpu Unwarp is %d\n", result_gpu);

    result_gpu == result_cpu ? printf("The result from gpu Unwarp is same with cpu\n") : printf("The result from gpu Unwarp isn't same with cpu\n");
    
    result_gpu = 0;
    CUDACHECK(reduceKernelUnwarpUroll8WithCuda<int>(in, &result_gpu, size));

    // printf("the result of cpu is %d\n", result_cpu);
    // printf("the result of gpu unwarp Unroll8 is %d\n", result_gpu);

    result_gpu == result_cpu ? printf("The result from gpu Unwarp Unroll8 is same with cpu\n") : printf("The result from gpu Unroll8 isn't same with cpu\n");

    result_gpu = 0;
    CUDACHECK(reduceKernelUnwarpUroll8WithCudaComplete<int>(in, &result_gpu, size));

    // printf("the result of cpu is %d\n", result_cpu);
    // printf("the result of gpu unwarp Unroll8 Complete is %d\n", result_gpu);

    result_gpu == result_cpu ? printf("The result from gpu Unwarp Unroll8 Complete is same with cpu\n") : printf("The result from gpu Unroll8 Complete isn't same with cpu\n");


    result_gpu = 0;
    CUDACHECK(reduceRecuriveKernelWithCuda<int>(in, &result_gpu, size));

    printf("the result of cpu is %d\n", result_cpu);
    printf("the result of gpu Recurive is %d\n", result_gpu);

    result_gpu == result_cpu ? printf("The result from gpu Recurive is same with cpu\n") : printf("The result from gpu Recurive isn't same with cpu\n");


    return 0;

}


