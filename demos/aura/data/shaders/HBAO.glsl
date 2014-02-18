// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------


-- CS


// =========================================
//  UNIFORMS 
// =========================================

uniform  vec2 uAOResolution;
uniform  vec2 uInvAOResolution;
uniform  vec2 uUVToViewA;
uniform  vec2 uUVToViewB;
uniform  float uR2;
uniform  float uTanAngleBias;
uniform  float uStrength; // only HBAO Y


uniform sampler2D uTexLinDepth;                       // input pass 1 & 2
writeonly uniform layout(r32f)  image2D uImgOutputX;  // output pass 1
readonly  uniform layout(r32f)  image2D uImgAOX;      // input pass 2
writeonly uniform layout(rg16f) image2D uImgOutputY;  // output pass 2

//------------------------------------------------------------------------------

// =========================================
//  STATIC PARAMETERS
// =========================================

// Step size in number of pixels
#ifndef STEP_SIZE
# define STEP_SIZE           4
#endif

// Number of shared-memory samples per direction
#ifndef NUM_STEPS
# define NUM_STEPS           8
#endif

// Maximum kernel radius in number of pixels
#ifndef KERNEL_RADIUS
# define KERNEL_RADIUS       (NUM_STEPS*NUM_STEPS)
#endif

// The last sample has weight = exp(-KERNEL_FALLOFF)
#define KERNEL_FALLOFF      3.0f

//
#ifndef HBAO_TILE_WIDTH
#define HBAO_TILE_WIDTH     320   // BLOCK_DIM
#endif

//------------------------------------------------------------------------------

// =========================================
//  SHARED MEMORY
// =========================================

#define SMEM_SIZE           (KERNEL_RADIUS + HBAO_TILE_WIDTH + KERNEL_RADIUS)
shared vec2 smem[SMEM_SIZE];
#define SMEM(x)             smem[(x)]

//------------------------------------------------------------------------------

// =========================================
//  FUNCTIONS
// =========================================

float tangent(vec2 v)
{
# define EPSILON 1.e-6f
  return -v.y / (abs(v.x) + EPSILON);
# undef  EPSILON
}

//------------------------------------------------------------------------------
float tanToSin(float x)
{
  return x * inversesqrt(x*x + 1.0f);
}

//------------------------------------------------------------------------------
float falloff(float sampleId)
{
  float r = sampleId / (NUM_STEPS - 1);
  return exp(-KERNEL_FALLOFF*r*r);
}

//------------------------------------------------------------------------------
vec2 minDiff(vec2 p, vec2 pr, vec2 pl)
{
  vec2 v1 = pr - p;
  vec2 v2 = p - pl;
  return (dot(v1,v1) < dot(v2,v2)) ? v1 : v2;
}

//------------------------------------------------------------------------------
void integrateDirection(inout float ao,
                        vec2 p, float tanT, int threadId, ivec2 stepSize)
{
  float tanH = tanT;
  float sinH = tanToSin(tanH);
  float sinT = tanToSin(tanT);

# pragma unroll
  // Per-sample attenuation
  for (int sampleId = 0; sampleId < NUM_STEPS; ++sampleId)
  {
    vec2 s = SMEM(threadId + sampleId*stepSize.y + stepSize.x);
    vec2 v = s - p;
    float tanS = tangent(v);
    float d2 = dot(v, v);

    if ((d2 < uR2) && (tanS > tanH))
    {
      // Accumulate AO between the horizon and the sample
      float sinS = tanToSin(tanS);
      ao += falloff(sampleId) * (sinS - sinH);

      // Update the current horizon angle
      tanH = tanS;
      sinH = sinS;
    }
  }
}

//------------------------------------------------------------------------------
// Bias tangent angle and compute HBAO in the +/- X or  +/- Y directions.
//------------------------------------------------------------------------------
float computeHBAO(vec2 p, vec2 t, int centerId)
{
  float ao = 0.0f;
  float tanT = tangent(t);
  ivec2 stepSize = ivec2(STEP_SIZE, STEP_SIZE);
  integrateDirection(ao, p, uTanAngleBias + tanT, centerId, +stepSize);
  integrateDirection(ao, p, uTanAngleBias - tanT, centerId, -stepSize);
  return ao;
}


//------------------------------------------------------------------------------
// Compute (X, Z) view-space coordinates from the depth texture
//------------------------------------------------------------------------------
vec2 fetchXZ(int x, int y)
{
  vec2 uv = (vec2(x, y) + 0.5f) * uInvAOResolution;
  float z_eye = texture(uTexLinDepth, uv).r;
  float x_eye = (uUVToViewA * uv.x + uUVToViewB) * z_eye;
  return vec2(x_eye, z_eye);
}

//------------------------------------------------------------------------------
// Compute (Y, Z) view-space coordinates from the depth texture
//------------------------------------------------------------------------------
vec2 fetchYZ(int x, int y)
{
  vec2 uv = (vec2(x, y) + 0.5f) * uInvAOResolution;
  float z_eye = texture(uTexLinDepth, uv).r;
  float y_eye = (uUVToViewA * uv.y + uUVToViewB) * z_eye;
  return vec2(y_eye, z_eye);
}

//------------------------------------------------------------------------------

// =========================================
//  HBAO SUBROUTINES
// =========================================
subroutine void HBAOFunction(ivec3 threadIdx, ivec3 blockIdx);

subroutine uniform HBAOFunction suHBAO;

//------------------------------------------------------------------------------
// Compute HBAO for the left and right directions
//------------------------------------------------------------------------------
subroutine(HBAOFunction)
void HBAO_X(ivec3 threadIdx, ivec3 blockIdx)
{
  const int tileStart  = blockIdx.x * HBAO_TILE_WIDTH;
  const int tileEnd    = tileStart + HBAO_TILE_WIDTH;
  const int apronStart = tileStart - KERNEL_RADIUS;
  const int apronEnd   = tileStart + KERNEL_RADIUS;

  const int x = apronStart + threadIdx.x;
  const int y = blockIdx.y;

  // Initialize shared memory tile
  SMEM(threadIdx.x) = fetchXZ(x, y);
  SMEM(min(2*KERNEL_RADIUS +threadIdx.x, SMEM_SIZE-1)) = fetchXZ(2*KERNEL_RADIUS+x, y);
  
  /*-------*/ barrier(); /*------------*/ // memoryBarrier();

  const ivec2 threadPos = ivec2(tileStart + threadIdx.x, blockIdx.y);  
  const int tileEndClamped = min(tileEnd, int(uAOResolution.x));

  if (threadPos.x < tileEndClamped)
  {
    int centerId = threadIdx.x + KERNEL_RADIUS;

    vec2 p  = SMEM(centerId);
    vec2 pr = SMEM(centerId+1);
    vec2 pl = SMEM(centerId-1);

    // Compute tangent vector using central differences
    vec2 t = minDiff(p, pr, pl);
    
    float ao = computeHBAO(p, t, centerId);
    imageStore(uImgOutputX, threadPos, ao.xxxx);
  }
}

//------------------------------------------------------------------------------
// Compute HBAO for the up and down directions.
// Output the average AO for the 4 axis-aligned directions.
//------------------------------------------------------------------------------
subroutine(HBAOFunction)
void HBAO_Y(ivec3 threadIdx, ivec3 blockIdx)
{
  const int   tileStart  = blockIdx.x * HBAO_TILE_WIDTH;
  const int   tileEnd    = tileStart + HBAO_TILE_WIDTH;
  const int   apronStart = tileStart - KERNEL_RADIUS;
  const int   apronEnd   = tileStart + KERNEL_RADIUS;

  const int x = blockIdx.y;
  const int y = apronStart + threadIdx.x;

  // Initialize shared memory tile
  SMEM(threadIdx.x) = fetchYZ(x, y);
  SMEM(min(2*KERNEL_RADIUS+threadIdx.x, SMEM_SIZE-1)) = fetchYZ(x, 2*KERNEL_RADIUS+y);

  /*-------*/ barrier(); /*------------*/ // memoryBarrier();

  const ivec2 threadPos = ivec2(blockIdx.y, tileStart + threadIdx.x);
  const int tileEndClamped = min(tileEnd, int(uAOResolution.y));

  if (threadPos.y < tileEndClamped)
  {
    int centerId = threadIdx.x + KERNEL_RADIUS;

    vec2 p  = SMEM(centerId);
    vec2 pt = SMEM(centerId + 1);
    vec2 pb = SMEM(centerId - 1);

    // Compute tangent vector using central differences
    vec2 t = minDiff(p, pt, pb);

    float aoy = computeHBAO(p, t, centerId);
    float aox = imageLoad(uImgAOX, threadPos ).r;
    float ao = (aox + aoy) * 0.25f;

    ao = clamp(1.0f- ao*uStrength, 0.0f, 1.0f);
    imageStore(uImgOutputY, threadPos, vec4(ao, p.y, 0.0f, 0.0f));
  }
}

//------------------------------------------------------------------------------

// =========================================
//  MAIN
// =========================================

layout(local_size_x = HBAO_TILE_WIDTH) in;

void main()
{
  const ivec3 threadIdx = ivec3(gl_LocalInvocationID);
  const ivec3 blockIdx  = ivec3(gl_WorkGroupID);
  suHBAO(threadIdx, blockIdx);
}
