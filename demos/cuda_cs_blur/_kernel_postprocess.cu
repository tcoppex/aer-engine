// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------


#include "cuda_cs_blur/kernel_postprocess.cuh"
#include "cuda_cs_blur/cuda_tools.h"


// TODO : try CUDA 5.0+ TextureObject / SurfaceObject

static
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> g_texSrcRef;

static
surface<void, cudaSurfaceType2D> g_surfDstRef;




__device__ 
int clamp(int x, int a, int b) {
    return max(a, min(b, x));
}

__device__ 
uchar4 clampResult(int r, int g, int b) {
    r = clamp(r, 0, 255);
    g = clamp(g, 0, 255);
    b = clamp(b, 0, 255);
    return make_uchar4( r , g, b, 255);
}

__global__
void kernel_postProcess(const uint imageWidth,
                        const uint tileWidth,
                        const int r)
{
  extern __shared__ uchar4 s_tile[];

# define SMEM(X, Y)         s_tile[(Y)*tileWidth + (X)]
# define SETPIXEL(X, Y)     tex2D( g_texSrcRef, (X), (Y))


  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bw = blockDim.x;
  int bh = blockDim.y;
  int x = blockIdx.x*bw + tx;
  int y = blockIdx.y*bh + ty;
  int r2 = r*r;


  SMEM(tx + r, ty + r) = SETPIXEL(x, y);

  // borders
  if (threadIdx.x < r)
  {
    // left
    SMEM(tx, ty + r) = SETPIXEL(x - r, y);
    // right
    SMEM(r + bw + tx, ty + r) = SETPIXEL(x + bw, y);
  }

  if (threadIdx.y < r)
  {
    // top
    SMEM(tx + r, ty) = SETPIXEL(x, y - r);
    // bottom
    SMEM(tx + r, r + bh + ty) = SETPIXEL(x, y + bh);
  }

  if ((threadIdx.x < r) && (threadIdx.y < r))
  {
    // tl
    SMEM(tx, ty) = SETPIXEL(x - r, y - r);
    // bl
    SMEM(tx, r + bh + ty) = SETPIXEL(x - r, y + bh);
    // tr
    SMEM(r + bw + tx, ty) = SETPIXEL(x + bh, y - r);
    // br
    SMEM(r + bw + tx, r + bh + ty) = SETPIXEL(x + bw, y + bh);
  }


  __syncthreads();


  // perform convolution
  int3 sum = make_int3(0,0,0);
  int samples = 0;

  for (int dy=-r; dy<=r; dy++)
  {
      for (int dx=-r; dx<=r; dx++)
      {

          uchar4 pixel = SMEM(r+tx+dx, r+ty+dy);

          // only sum pixels within disc-shaped kernel
          int l = dx*dx + dy*dy;

          if (l <= r2)
          {
              sum.x += pixel.x;
              sum.y += pixel.y;
              sum.z += pixel.z;
              
              ++samples;
          }
      }
  }

  sum.x /= samples;
  sum.y /= samples;
  sum.z /= samples;
  
  // ABGR
  uchar4 data = clampResult(sum.x, sum.y, sum.z);
  surf2Dwrite( data, g_surfDstRef, 4*x, y);
}



void launch_cuda_kernel( const dim3 gridDim, 
                         const dim3 blockDim, 
                         const size_t smemSize,
                         cudaArray *const d_in, 
                         cudaArray *const d_out,
                         const unsigned int imageWidth,
                         const unsigned int tileWidth,
                         const unsigned int radius)
{
  // Input texture bind to a CUDA Texture reference
  CHECK_CUDA( cudaBindTextureToArray( g_texSrcRef, d_in) );

  // Output texture bind to a CUDA Surface reference
  CHECK_CUDA( cudaBindSurfaceToArray( g_surfDstRef, d_out) );

  kernel_postProcess<<<gridDim, blockDim, smemSize>>>( imageWidth, tileWidth, radius);
  cudaDeviceSynchronize();


  CHECKCUDAERROR();
}
