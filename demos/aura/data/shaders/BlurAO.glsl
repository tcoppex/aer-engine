// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------


-- X.CS


// =========================================
//  UNIFORMS 
// =========================================

uniform  float uBlurFalloff;
uniform  float uBlurDepthThreshold;
uniform  vec2 uInvFullResolution;
uniform  vec2 uFullResolution;

uniform sampler2D uTexAOLinDepthNearest;
uniform sampler2D uTexAOLinDepthLinear;

writeonly uniform layout(rg16f) image2D uDstImg;


// =========================================
//  STATIC PARAMETERS
// =========================================

#define ROW_TILE_W          320

#ifndef KERNEL_RADIUS
# define KERNEL_RADIUS      8
#endif

// Should be specified at compile time
#ifndef BLOCK_DIM
# define BLOCK_DIM          (KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS)
#endif

#define SMEM_SIZE           BLOCK_DIM


// =========================================
//  CONSTANTS
// =========================================

const int kBlurRadius     = KERNEL_RADIUS;
const int kHalfBlurRadius = kBlurRadius / 2;

// =========================================
//  SHARED MEMORY
// =========================================

shared vec2 smem[SMEM_SIZE];

// =========================================
//  MACROS
// =========================================

#define TEX_AOZ_NEAREST(uv)   texture( uTexAOLinDepthNearest, (uv)).rg
#define TEX_AOZ_LINEAR(uv)    texture( uTexAOLinDepthLinear,  (uv)).rg
#define SMEM(x)               smem[(x)]

// =========================================
//  FUNCTIONS
// =========================================

float CrossBilateralWeight(float r, float d, float d0)
{
  // The exp2(-r*r*g_BlurFalloff) expression below is pre-computed by fxc.
  // On GF100, this ||d-d0||<threshold test is significantly faster than 
  // an exp weight.
  return exp2(-r*r*uBlurFalloff) * float(abs(d - d0) < uBlurDepthThreshold);
}

void blurX(const ivec3 threadIdx, const ivec3 blockIdx)
{
  const float        row = float(blockIdx.y);
  const float  tileStart = float(blockIdx.x * ROW_TILE_W);
  const float    tileEnd = tileStart + ROW_TILE_W;
  const float apronStart = tileStart - KERNEL_RADIUS;
  const float   apronEnd =   tileEnd + KERNEL_RADIUS;

  // Fetch (ao, z) vetween adjacent pixels with linear interpolation
  const float x = apronStart + float(threadIdx.x) + 0.5f;
  const float y = row;
  const vec2 uv = (vec2(x,y) + 0.5f) * uInvFullResolution;
  
  SMEM(threadIdx.x) = TEX_AOZ_LINEAR(uv);

  /*----------*/ barrier(); /*----------*/

  const float writePos = tileStart + threadIdx.x;
  const float tileEndClamped = min( tileEnd, uFullResolution.x);

  if (writePos < tileEndClamped)
  {
    // Fetch (ao, z) at the kernel center
    vec2 uv = (vec2(writePos, y) + 0.5f) * uInvFullResolution;
    vec2 aoDepth = TEX_AOZ_NEAREST(uv);
    float ao_total = aoDepth.x;
    float center_d = aoDepth.y;
    float w_total = 1.0f;

#   pragma unroll
    for (int i=0; i<kHalfBlurRadius; ++i)
    {
      // Sample the pre-filtered data with step size = 2 pixels
      float r = 2.0f*i + (0.5f-kBlurRadius);
      uint j = 2u*i + threadIdx.x;
      vec2 samp = SMEM(j);
      float w = CrossBilateralWeight( r, samp.y, center_d);
      ao_total += w * samp.x;
      w_total  += w;
    }

#   pragma unroll
    for (int i=0; i<kHalfBlurRadius; ++i)
    {
      // Sample the pre-filtered data with step size = 2 pixels
      float r = 2.0f*i + 1.5f;
      uint j = 2*i + threadIdx.x + KERNEL_RADIUS + 1;
      vec2 samp = SMEM(j);
      float w = CrossBilateralWeight( r, samp.y, center_d);
      ao_total += w * samp.x;
      w_total  += w;
    }

    float ao = ao_total / w_total;
    vec4 output = vec4( ao, center_d, 0.0f, 0.0f);
    imageStore( uDstImg, ivec2(writePos, blockIdx.y), output);
  }
}


// =========================================
//  MAIN
// =========================================
layout(local_size_x = BLOCK_DIM) in;

void main()
{
  const ivec3 threadIdx = ivec3(gl_LocalInvocationID);
  const ivec3 blockIdx  = ivec3(gl_WorkGroupID);
  blurX( threadIdx, blockIdx);
}

--


// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------



-- Y.CS


// =========================================
//  UNIFORMS 
// =========================================

uniform  float uBlurFalloff;
uniform  float uBlurDepthThreshold;
uniform  vec2 uInvFullResolution;
uniform  vec2 uFullResolution;

uniform sampler2D uTexAOLinDepthNearest;
uniform sampler2D uTexAOLinDepthLinear;

writeonly uniform layout(r32f) image2D uDstImg;


// =========================================
//  STATIC PARAMETERS
// =========================================

#define COL_TILE_W          320

#ifndef KERNEL_RADIUS
# define KERNEL_RADIUS      8
#endif 

#ifndef BLOCK_DIM
# define BLOCK_DIM          (KERNEL_RADIUS + COL_TILE_W + KERNEL_RADIUS)
#endif

#define SMEM_SIZE           BLOCK_DIM



// =========================================
//  CONSTANTS
// =========================================

const int kBlurRadius     = KERNEL_RADIUS;
const int kHalfBlurRadius = kBlurRadius / 2;

// =========================================
//  SHARED MEMORY
// =========================================

shared vec2 smem[SMEM_SIZE];

// =========================================
//  MACROS
// =========================================

#define TEX_AOZ_NEAREST(uv)   texture( uTexAOLinDepthNearest, (uv), 0).rg
#define TEX_AOZ_LINEAR(uv)    texture( uTexAOLinDepthLinear,  (uv), 0).rg
#define SMEM(x)               smem[(x)]

// =========================================
//  FUNCTION
// =========================================

float CrossBilateralWeight(float r, float d, float d0)
{
  // The exp2(-r*r*g_BlurFalloff) expression below is pre-computed by fxc.
  // On GF100, this ||d-d0||<threshold test is significantly faster than 
  // an exp weight.
  return exp2(-r*r*uBlurFalloff) * float(abs(d - d0) < uBlurDepthThreshold);
}

// =========================================
//  MAIN
// =========================================

layout(local_size_x = BLOCK_DIM) in;

void main()
{
  const ivec3 threadIdx = ivec3(gl_LocalInvocationID);
  const ivec3 blockIdx  = ivec3(gl_WorkGroupID);

  const float        col = float(blockIdx.y);
  const float  tileStart = float(blockIdx.x) * COL_TILE_W;
  const float    tileEnd = tileStart + COL_TILE_W;
  const float apronStart = tileStart - KERNEL_RADIUS;
  const float   apronEnd = tileEnd + KERNEL_RADIUS;

  const float x = col;
  const float y = apronStart + float(threadIdx.x) + 0.5f;
  const vec2 uv = (vec2(x, y) + 0.5f) * uInvFullResolution; //
  SMEM(threadIdx.x) = TEX_AOZ_LINEAR(uv);

  /*-------*/ barrier(); /*-------*/

  const float writePos = tileStart + float(threadIdx.x);
  const float tileEndClamped = min( tileEnd, uFullResolution.x);

  if (writePos < tileEndClamped)
  {
    vec2 uv = (vec2( x, writePos) + 0.5f) * uInvFullResolution;//
    vec2 aoDepth = TEX_AOZ_NEAREST(uv);
    float ao_total = aoDepth.x;
    float center_d = aoDepth.y;
    float w_total = 1.0f;

#   pragma unroll
    for (int i=0; i<kHalfBlurRadius; ++i)
    {
      float r = 2.0f*i + (-kBlurRadius + 0.5f);
      uint j = 2u*i + threadIdx.x;
      vec2 samp = SMEM(j);
      float w = CrossBilateralWeight( r, samp.y, center_d);
      ao_total += w * samp.x;
      w_total += w;
    }

#   pragma unroll
    for (int i=0; i<kHalfBlurRadius; ++i)
    {
      float r = 2.0f * i + 1.5f;
      uint j = 2u*i + threadIdx.x + KERNEL_RADIUS + 1u;
      vec2 samp = SMEM(j);
      float w = CrossBilateralWeight( r, samp.y, center_d);
      ao_total += w * samp.x;
      w_total += w;
    }

    float ao = ao_total / w_total;
    imageStore( uDstImg, ivec2( blockIdx.y, writePos), ao.xxxx);
  }
}
