#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../My_CUDA.hpp"
#include <iostream>



#define SIZE 10


#define check(call) \
{\
 const cudaError_t error = call;\
 if (error != cudaSuccess) {\
    printf("Error occurs' location is %s : %d\n", __FILE__, __LINE__); \
    printf("Error occurs, Reason is : %s , and Codo is : %d\n", cudaGetErrorString(error) , error);\
    exit(1); \
 }\
}

// bool check(cudaError_t status) {
//     if (status != cudaSuccess) {
//         printf("Error occurs, Reason is : %s , and Codo is : %d", cudaGetErrorString(status) , status);
//         return true;
//     }
//     return false;
// }


__global__ void helloworld() {
    printf("Hello World, CUDA!\n");
}


__global__ void add(int* in1, int* in2, int* out) {
    int index = threadIdx.x;
    out[index] = in1[index] + in2[index];
}

int main(void) {
    
    cudaError_t status;
    int* in1_h = static_cast<int*>(malloc(SIZE * sizeof(int)));
    int* in2_h = static_cast<int*>(malloc(SIZE * sizeof(int)));
    for (int i = 0; i < SIZE; ++i) {
        in1_h[i] = i;
        in2_h[i] = i;
    }

    int* in1_d = 0;
    int* in2_d = 0;
    status = cudaMalloc((void**)&in1_d, SIZE * sizeof(int));
    CUDACHECK(status);
    status = cudaMalloc((void**)&in2_d, SIZE * sizeof(int));
    check(status);
    status = cudaMemcpy(in1_d, in1_h, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    check(status);
    status = cudaMemcpy(in2_d, in2_h, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    check(status);

    int* out_d = 0;
    status = cudaMalloc((void**)&out_d, SIZE * sizeof(int));
    check(status);
     

    dim3 grid(1);
    dim3 block(SIZE);
    add <<< grid, block >>>(in1_d, in2_d, out_d);
    
    status = cudaDeviceSynchronize();
    check(status);
    
    int* out_h = static_cast<int*>(malloc(SIZE*sizeof(int)));
    cudaMemcpy(out_h, out_d, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

    // output
    printf("in1 : \n");
    for(int i = 0; i < SIZE; ++i) {
        printf("%d  ", in1_h[i]);
    }
    printf("\n");

    printf("in2 : \n");
    for(int i = 0; i < SIZE; ++i) {
        printf("%d  ", in2_h[i]);
    }
    printf("\n");

    printf("out : \n");
    for(int i = 0; i < SIZE; ++i) {
        printf("%d  ", out_h[i]);
    }
    printf("\n");


    helloworld<<<1, 10>>>();

Error:
    cudaFree(in1_d);
    cudaFree(in2_d);
    cudaFree(out_d);
    cudaDeviceReset();




}