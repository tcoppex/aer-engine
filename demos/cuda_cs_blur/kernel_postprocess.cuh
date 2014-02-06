// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef CUDAGLCOMPUTE_CUDAPOSTPROCESS_CUH
#define CUDAGLCOMPUTE_CUDAPOSTPROCESS_CUH

// Host Kernel launcher -------------------------------------------------------

void launch_cuda_kernel(const dim3 gridDim, 
                        const dim3 blockDim,
                        const size_t smemSize,
                        cudaArray *const d_in,
                        cudaArray *const d_out,
                        const unsigned int imageWidth,
                        const unsigned int tileWidth,
                        const unsigned int radius);


#endif //CUDAGLCOMPUTE_CUDAPOSTPROCESS_CUH
