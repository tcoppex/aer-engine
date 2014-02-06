// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef CUDA_CS_BLUR_CUDA_TOOLS_H_
#define CUDA_CS_BLUR_CUDA_TOOLS_H_

#include <cuda_runtime.h>


// Utility function to handle CUDA error --------------------------------------

#include <iostream>

template<typename T>
void checkCUDA(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

#define CHECK_CUDA(foo)   checkCUDA((foo), #foo, __FILE__, __LINE__)
#define CHECKCUDAERROR()  CHECK_CUDA(cudaGetLastError())


// Utility structure to record GPU time ---------------------------------------

struct CUDATimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  CUDATimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~CUDATimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() {
    cudaEventRecord(start, 0);
  }

  void Stop() {
    cudaEventRecord(stop, 0);
  }

  float Elapsed() {
    float elapsed(0.0f);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

#endif // CUDA_CS_BLUR_CUDA_TOOLS_H_
