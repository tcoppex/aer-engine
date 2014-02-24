//
//      Compute Shader to compute the blur of an image
//
// Process on RG16f textures
//


-- CS


// -- STATIC PARAMETERS

#ifndef BLOCK_DIM
# define BLOCK_DIM      16
#endif

#define RADIUS          4

#ifndef MAX_RADIUS
# define MAX_RADIUS     RADIUS//BLOCK_DIM
#endif

#define TILE_MAXDIM     (BLOCK_DIM + 2*MAX_RADIUS)
#define SMEM_SIZE       (TILE_MAXDIM*TILE_MAXDIM)

layout(local_size_x=BLOCK_DIM, local_size_y=BLOCK_DIM) in;


// -- PARAMETERS

// Using a sampler as source instead of an image allow for texture clamping
uniform sampler2D uSrcTex;

writeonly 
uniform layout(rg16f) image2D uDstImg;


// -- CONSTANTS

const vec2 kTexelSize = 1.0f / textureSize(uSrcTex, 0).xy;

// Note: not on constant register due to RADIUS use
const int  kTileDim   = BLOCK_DIM + 2*RADIUS;


// -- SHARED MEMORY
  
// Tile used for the blur kernel computation.
// Pixels value (RG16F) are stored as uint (2*short) to save memory.
shared uint s_tile[SMEM_SIZE];


// -- MACROS

// Easy access to shared memory
#define SMEM(X, Y)       s_tile[(Y)*kTileDim+(X)]
// Load texel from unormalize texture coordinates
#define LOADTEXEL(X, Y)  texture(uSrcTex, ivec2(X,Y)*kTexelSize, 0).rg
// Pack a pixel from src into an uint to store in shared memory
#define SETPIXEL(X, Y)   packHalf2x16(LOADTEXEL(X, Y))
// Convert back a pixel from shared memory uint to vec2
#define GETPIXEL(X, Y)   unpackHalf2x16(SMEM(X,Y)).xy


// -- MAIN

void main()
{
  // -- IO functions used signed integers (renamed as CUDA) --
  const ivec3 gridDim   = ivec3(gl_NumWorkGroups);
  const ivec3 blockDim  = ivec3(gl_WorkGroupSize);
  const ivec3 blockIdx  = ivec3(gl_WorkGroupID);
  const ivec3 threadPos = ivec3(gl_GlobalInvocationID);
  const ivec3 threadIdx = ivec3(gl_LocalInvocationID);
  //const int   tid       =   int(gl_LocalInvocationIndex);


  // Simplify writing
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bw = blockDim.x;
  const int bh = blockDim.y;
  const int x = threadPos.x;
  const int y = threadPos.y;
  const int r = RADIUS;


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


  /* ------- */ barrier(); /* ------- */

  const int r2 = r*r;
  vec4 sum = vec4(0.0f);
  int samples = 0;

  for (int dy=-r; dy<=r; ++dy)
  {
    for (int dx=-r; dx<=r; ++dx)
    {
      // only sum pixels within disc-shaped kernel
      float l = dx*dx + dy*dy;

      if (l <= r2)
      {
        vec2 pixel = GETPIXEL(tx+r+dx, ty+r+dy);

        sum.xy += pixel;
        ++samples;
      }
    }
  }

  sum /= samples;
  imageStore(uDstImg, threadPos.xy, sum);
}
