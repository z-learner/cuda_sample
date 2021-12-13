#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <stdlib.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

#define TILE_DIM 32
#define BLOCK_DIM 32
static bool getError(cudaError_t cudaStatus, char* message);

cudaError_t transposeWithCuda(float* odata, float* idata, int width, int height);
cudaError_t transposeSharedMemWithCuda(float* odata, float* idata, int width, int height);
cudaError_t transposeSharedMemNoBankingWithCuda(float* odata, float* idata, int width, int height);




__global__ void transposeNaive(float* odata, float* idata, int width, int height) {
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int in_index = xIndex + width * yIndex;
    int out_index = yIndex + height * xIndex;

    odata[out_index] = idata[in_index];
}

__global__ void transposeSharedMem(float* odata, float* idata, int width, int height) {
    __shared__ float shMat[TILE_DIM][TILE_DIM];
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int in_index = xIndex + width * yIndex;
    int out_index = yIndex + height * xIndex;
    shMat[threadIdx.x][threadIdx.y] = idata[in_index];
    __syncthreads();
    odata[out_index] = shMat[threadIdx.x][threadIdx.y];
}

__global__ void transposeSharedMemNoBanking(float* odata, float* idata, int width, int height) {
    __shared__ float shMat[TILE_DIM][TILE_DIM + 1];
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int in_index = xIndex + width * yIndex;
    int out_index = yIndex + height * xIndex;
    shMat[threadIdx.x][threadIdx.y] = idata[in_index];
    __syncthreads();
    odata[out_index] = shMat[threadIdx.x][threadIdx.y];
}




int main()
{
    
    srand((int)time(0));

    int width = TILE_DIM * 16;
    int heigh = TILE_DIM * 8;

    MatrixXf midata = MatrixXf::Zero(heigh, width); // row = heigh  col = width
    auto modata = midata.transpose();


    float* indata = static_cast<float*>(malloc(width * heigh * sizeof(float)));
    for (int i = 0; i < width * heigh; ++i) {
        indata[i] = (float)rand() / (float)INT32_MAX;
        midata(i / width, i % width) = indata[i];
    }
    float* odata = static_cast<float*>(malloc(width * heigh * sizeof(float)));


    cudaError_t cudaStatus = transposeWithCuda(odata, indata, width, heigh);
    if (getError(cudaStatus, "transposeWithCuda failed")) return 1;
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (getError(cudaStatus, "cudaDeviceReset failed")) return 1;

    // Test
    for (int i = 0; i < width * heigh; ++i) {
        if (modata(i / heigh, i % heigh) != odata[i]) {
            cout << "transposeWithCuda is wrong" << endl;
            goto transposeSharedMemWithCudaTest;
        }
    }
    cout << "transposeWithCuda succeed" << endl;

transposeSharedMemWithCudaTest:

    cudaStatus = transposeSharedMemWithCuda(odata, indata, width, heigh);
    if (getError(cudaStatus, "transposeSharedMemWithCuda failed")) return 1;
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (getError(cudaStatus, "cudaDeviceReset failed")) return 1;

    for (int i = 0; i < width * heigh; ++i) {
        if (modata(i / heigh, i % heigh) != odata[i]) {
            cout << "transposeSharedMemWithCuda is wrong" << endl;
            goto transposeSharedMemNoBankingWithCuda;
        }
    }
    cout << "transposeSharedMemWithCuda succeed" << endl;



transposeSharedMemNoBankingWithCuda:
    cudaStatus = transposeSharedMemNoBankingWithCuda(odata, indata, width, heigh);
    if (getError(cudaStatus, "transposeSharedMemNoBankingWithCuda failed")) return 1;
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (getError(cudaStatus, "cudaDeviceReset failed")) return 1;

    for (int i = 0; i < width * heigh; ++i) {
        if (modata(i / heigh, i % heigh) != odata[i]) {
            cout << "transposeSharedMemNoBankingWithCuda is wrong" << endl;
            goto End;
        }
    }
    cout << "transposeSharedMemNoBankingWithCuda succeed" << endl;


End:
    return 0;
}

static bool getError(cudaError_t cudaStatus, char* message) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, message);
        return true;
    }
    return false;
}



cudaError_t transposeSharedMemNoBankingWithCuda(float* odata, float* idata, int width, int height) {
    float* dev_odata;
    float* dev_indata;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    

    cudaStatus = cudaMalloc((void**)&dev_indata, width * height * sizeof(float));
    

    cudaStatus = cudaMalloc((void**)&dev_odata, width * height * sizeof(float));
    

    cudaStatus = cudaMemcpy(dev_indata, idata, height * width * sizeof(float), cudaMemcpyHostToDevice);
    

    dim3 dimGrid(width / TILE_DIM, height / TILE_DIM);
    dim3 dimBlock(TILE_DIM, TILE_DIM);

    transposeSharedMemNoBanking << < dimGrid, dimBlock >> > (dev_odata, dev_indata, width, height);
    cudaStatus = cudaGetLastError();


    // wait all done
    cudaStatus = cudaDeviceSynchronize();

    // copy back
    cudaStatus = cudaMemcpy(odata, dev_odata, width * height * sizeof(float), cudaMemcpyDeviceToHost);



    cudaFree(dev_indata);
    cudaFree(dev_odata);

    return cudaStatus;
}


cudaError_t transposeSharedMemWithCuda(float* odata, float* idata, int width, int height) {
    float* dev_odata;
    float* dev_indata;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    

    cudaStatus = cudaMalloc((void**)&dev_indata, width * height * sizeof(float));
    

    cudaStatus = cudaMalloc((void**)&dev_odata, width * height * sizeof(float));
   

    cudaStatus = cudaMemcpy(dev_indata, idata, height * width * sizeof(float), cudaMemcpyHostToDevice);


    dim3 dimGrid(width / TILE_DIM, height / TILE_DIM);
    dim3 dimBlock(TILE_DIM, TILE_DIM);

    transposeSharedMem << < dimGrid, dimBlock >> > (dev_odata, dev_indata, width, height);
    cudaStatus = cudaGetLastError();


    // wait all done
    cudaStatus = cudaDeviceSynchronize();


    // copy back
    cudaStatus = cudaMemcpy(odata, dev_odata, width * height * sizeof(float), cudaMemcpyDeviceToHost);



Error:
    cudaFree(dev_indata);
    cudaFree(dev_odata);

    return cudaStatus;
}


cudaError_t transposeWithCuda(float* odata, float* idata, int width, int height) {
    float* dev_odata;
    float* dev_indata;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);


    cudaStatus = cudaMalloc((void**)&dev_indata, width * height * sizeof(float));

    cudaStatus = cudaMalloc((void**)&dev_odata, width * height * sizeof(float));


    cudaStatus = cudaMemcpy(dev_indata, idata, height * width * sizeof(float), cudaMemcpyHostToDevice);


    dim3 dimGrid(width / TILE_DIM, height / TILE_DIM);
    dim3 dimBlock(TILE_DIM, TILE_DIM);

    transposeNaive << < dimGrid, dimBlock >> > (dev_odata, dev_indata, width, height);
    cudaStatus = cudaGetLastError();


    // wait all done
    cudaStatus = cudaDeviceSynchronize();


    // copy back
    cudaStatus = cudaMemcpy(odata, dev_odata, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_indata);
    cudaFree(dev_odata);

    return cudaStatus;
}