#include <iostream>
#include <cstdio>
#include "cuda_runtime.h"
#include <thread>
#include <vector>
#include <sys/time.h>
using namespace std;

#define CUDACHECK(call) \
{\
 const cudaError_t error = call;\
 if (error != cudaSuccess) {\
    printf("Error occurs, location is %s : %d\n", __FILE__, __LINE__); \
    printf("Error occurs, Reason is : %s , and Code is : %d\n", cudaGetErrorString(error) , error);\
    exit(1); \
 }\
}

// get current time
double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void setValueKernel(int* data, int& value, unsigned int size) {
    int idx = blockDim.x + blockIdx.x + threadIdx.x;
    if (idx < size) data[idx] = value;
}

// message
class cuda_message {
public:
    int cudaDeviceId;
    cudaStream_t stream;
    int id;
    int* data_h;
    int* data_d;
    unsigned int data_size;
    bool success = false ;
};


// check result 
bool postprocess(void *void_arg) {
    cuda_message* workload = static_cast<cuda_message*>(void_arg);
    printf("Stream : %d start to check result.\n", workload->id);

    // CUDACHECK(cudaSetDevice(workload->cudaDeviceId));

    workload->success = true;
    for (int i = 0; i < workload->data_size; ++i) workload->success &= (workload->data_h[i] == 2 * workload->id);
    
    return workload->success;

}

// mycallback function
void CUDART_CB myCallBack(cudaStream_t stream, cudaError_t status, void *arg) {
    cuda_message* workload = static_cast<cuda_message*>(arg);
    printf("Stream : %d calls callback successfully.\n", workload->id);
    // CUDACHECK(status);
    postprocess(arg);
    printf("Stream : %d checks result successfully.\n", workload->id);
}

// the function pre thread
void lunch(void* arg) {

    cuda_message* workload = static_cast<cuda_message*>(arg);
    printf("thread %d lunch successfully.\n", workload->id);
    CUDACHECK(cudaSetDevice(workload->cudaDeviceId));

    // Allocate
    CUDACHECK(cudaStreamCreate(&workload->stream));
    CUDACHECK(cudaMalloc(&workload->data_d, workload->data_size * sizeof(int)));
    CUDACHECK(cudaHostAlloc(&workload->data_h,  workload->data_size * sizeof(int), cudaHostAllocPortable));

    memset(workload->data_h, workload->id, workload->data_size * sizeof(int));

    // Schedule work for GPU in CUDA stream
    // Note: Dedicated streams enable concurrent execution of workloads on the GPU
    dim3 block(512);
    dim3 grid((workload->data_size + block.x - 1) / block.x);

    CUDACHECK(cudaMemcpyAsync(workload->data_d, workload->data_h, workload->data_size * sizeof(int), cudaMemcpyHostToDevice, workload->stream));
    setValueKernel<<<grid, block, 0, workload->stream>>>(workload->data_d, workload->id, workload->data_size);
    CUDACHECK(cudaMemcpyAsync(workload->data_h, workload->data_d, workload->data_size * sizeof(int), cudaMemcpyDeviceToHost, workload->stream));

    CUDACHECK( cudaStreamAddCallback(workload->stream, myCallBack, workload, 0) );
    // CUDACHECK( cudaStreamSynchronize(workload->stream) );
    return ;
}

int main() {
    double iStart, iElaps;
    int dev = 0;
    // simpleDeviceQuery(dev);
    CUDACHECK(cudaSetDevice(dev));
    unsigned int stream_number = 8;
    unsigned int size = 1 << 20;
    cuda_message *cuda_messages;
    cuda_messages = (cuda_message *) malloc(stream_number * sizeof(cuda_message));;
    std::vector<std::thread> threads(stream_number);

  
    for (int i = 0; i < stream_number; ++i) {
        cuda_messages[i].data_size = size;
        cuda_messages[i].id = i + 1;
        cuda_messages[i].cudaDeviceId = dev;
        cuda_messages[i].success = false;
        // std::cout << "New thread start to lunch" << std::endl;
        threads[i] = std::move(std::thread(lunch, cuda_messages + i));
        // std::cout << "New thread lunchs successfully" << std::endl;
    }
    // CUDACHECK( cudaDeviceSynchronize() );
    iStart = cpuSecond();
    iElaps = cpuSecond() - iStart;
    
    // wait until gpu done
    while (iElaps < 10)
    {
        iElaps = cpuSecond() - iStart;
    }
    

    for (int i = 0; i < stream_number; ++i) {
        // CUDACHECK(cudaStreamSynchronize(cuda_messages[i].stream));
        threads[i].join();
        cuda_messages[i].success ? printf("StramId : %d, Right.\n", cuda_messages[i].id) : printf("StramId : %d, Wrong.\n", cuda_messages[i].id);
        CUDACHECK(cudaFreeHost(cuda_messages[i].data_h));
        CUDACHECK(cudaFree(cuda_messages[i].data_d));
        CUDACHECK(cudaStreamDestroy(cuda_messages[i].stream));
    }

    free(cuda_messages);

}