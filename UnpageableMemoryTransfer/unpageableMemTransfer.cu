#include <iostream>
#include <stdio.h>
#include "My_CUDA.hpp"


using namespace std;



int main() {
    simpleDeviceQuery(0);

    unsigned int isize = 1 << 20;
    unsigned int nbyte = isize * sizeof(float);

    float* data_h = static_cast<float*>(malloc(nbyte));
    memset(data_h, 0, nbyte);


    float* data_d;
    CUDACHECK(cudaMallocHost((void**)&data_d, nbyte));

    CUDACHECK(cudaMemcpy(data_d, data_h, nbyte, cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemcpy(data_h, data_d, nbyte, cudaMemcpyDeviceToHost));


    CUDACHECK(cudaFreeHost(data_d));

    free(data_h);
}