/*
 *          Noise.glsl
 *
 *      Compute Classical Perlin Noise (2D/3D).
 *
 *      ref : 'Improving Noise' - Ken Perlin
 *            'Implementing Improved Perlin Noise' - Ken Perlin
 *            'Implementing Improved Perlin Noise' - Simon Green
 *            'Advanced Perlin Noise' - Inigo Quilez
 *            'Simplex noise demystified' - Stefan Gustavson
 *
 *      Note : This code is based on Stefan Gustavson & Ian McEwan noise 
 *             implementation.
 *
 *             This is not a MAIN shader, it must be included.
 *
 *      Todo : implement Simplex Noise.
 *
 *      version : GLSL 3.1+ core
 *
 *      profile : vertex / ? / fragment
 */
 

-- Include


uniform bool uEnableTiling = false;
uniform vec3 uTileRes = vec3(256.0f);

uniform int uPermutationSeed = 0;


//---------------------------- UTILS


// Fast computation of x modulo 289
vec3 mod289(vec3 x) {
  return x - floor(x * (1.0f / 289.0f)) * 289.0f;
}

vec4 mod289(vec4 x) {
  return x - floor(x * (1.0f / 289.0f)) * 289.0f;
}

// Compute indices for the PRNG
vec4 permute(vec4 x) {
  return mod289(((x*34.0f)+1.0f)*x + uPermutationSeed);
}

// Quintic interpolant
vec2 fade(vec2 u) {
  return u*u*u*(u*(u*6.0f - 15.0f) + 10.0f);
  
  // Original cubic interpolant (faster, but not 2nd order derivable)
  //return u*u*(3.0f - 2.0f*u);
}

vec3 fade(vec3 u) {
  return u*u*u*(u*(u*6.0f - 15.0f) + 10.0f);
}

float normalizeNoise(float n) {
  // return noise in [0, 1]
  return 0.5f*(2.44f*n + 1.0f);
}


//---------------------------- NOISE 2D

void pnoise_gradients(in  vec2 pt,
                      in  vec2 scaledTileRes,
                      out vec4 gradients,
                      out vec4 fpt) 
{
  // Retrieve the integral part (for indexation)
  vec4 ipt = floor(pt.xyxy) + vec4(0.0f, 0.0f, 1.0f, 1.0f);

  // Tile the noise (if enabled)
  if (uEnableTiling) {
    ipt = mod(ipt, scaledTileRes.xyxy);
  }
  ipt = mod289(ipt);

  // Compute the 4 corners hashed gradient indices
  vec4 ix = ipt.xzxz;
  vec4 iy = ipt.yyww;
  vec4 p = permute(permute(ix) + iy);
  /*
  Fast version for :
  p.x = P(P(ipt.x)      + ipt.y);
  p.y = P(P(ipt.x+1.0f) + ipt.y);
  p.z = P(P(ipt.x)      + ipt.y+1.0f);
  p.w = P(P(ipt.x+1.0f) + ipt.y+1.0f);
  */

  // With 'p', computes Pseudo Random Numbers
  const float one_over_41 = 1.0f / 41.0f; //0.02439f
  vec4 gx = 2.0f * fract(p * one_over_41) - 1.0f;
  vec4 gy = abs(gx) - 0.5f;
  vec4 tx = floor(gx + 0.5f);
  gx = gx - tx;

  // Create unnormalized gradients
  vec2 g00 = vec2(gx.x,gy.x);
  vec2 g10 = vec2(gx.y,gy.y);
  vec2 g01 = vec2(gx.z,gy.z);
  vec2 g11 = vec2(gx.w,gy.w);

  // 'Fast' normalization
  vec4 dp = vec4(dot(g00, g00), dot(g10, g10), dot(g01, g01), dot(g11, g11));
  vec4 norm = inversesqrt(dp);
  g00 *= norm.x;
  g10 *= norm.y;
  g01 *= norm.z;
  g11 *= norm.w;

  // Retrieve the fractional part (for interpolation)
  fpt = fract(pt.xyxy) - vec4(0.0f, 0.0f, 1.0f, 1.0f);

  // Calculate gradient's influence
  vec4 fx = fpt.xzxz;
  vec4 fy = fpt.yyww;
  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));
  /*
  Fast version for :
  n00 = dot(g00, fpt + vec2(0.0f, 0.0f));
  n10 = dot(g10, fpt + vec2(-1.0f, 0.0f));
  n01 = dot(g01, fpt + vec2(0.0f,-1.0f));
  n11 = dot(g11, fpt + vec2(-1.0f,-1.0f));
  */

  gradients = vec4(n00, n10, n01, n11);
} 

// Classical Perlin Noise 2D
float pnoise(vec2 pt, vec2 scaledTileRes = vec2(0.0f)) {
  vec4 g, fpt;
  pnoise_gradients(pt, scaledTileRes, g, fpt);

  // Interpolate gradients
  vec2 u = fade(fpt.xy);
  float n1 = mix(g.x, g.y, u.x);
  float n2 = mix(g.z, g.w, u.x);
  float noise = mix(n1, n2, u.y);

  return noise;
}


// Derivative Perlin Noise 2D
vec3 dpnoise(vec2 pt, vec2 scaledTileRes = vec2(0.0f)) {
  vec4 g, fpt;
  pnoise_gradients(pt, scaledTileRes, g, fpt);

  float k0 = g.x;
  float k1 = g.y - g.x;
  float k2 = g.z - g.x;
  float k3 = g.x - g.z - g.y + g.w;
  vec3 res;

  vec2 u = fade(fpt.xy);
  res.x = k0 + k1*u.x + k2*u.y + k3*u.x*u.y;

  vec2 dpt = 30.0f*fpt.xy*fpt.xy*(fpt.xy*(fpt.xy-2.0f)+1.0f);
  res.y = dpt.x * (k1 + k3*u.y);
  res.z = dpt.y * (k2 + k3*u.x);

  return res;
}


// -- Fractional Brownian Motion function --

/*  
 NOTE : Tiling does not works with non integer frequency or non power of two zoom.
*/

// Classical Perlin Noise fbm 2D
float fbm_pnoise(in vec2 pt, 
                 in float zoom,
                 const int numOctave, 
                 const float frequency, 
                 const float amplitude)
{
  float sum = 0.0f;
  float f = frequency;
  float w = amplitude;

  vec2 v = zoom * (pt);
  vec2 scaledTileRes = zoom * uTileRes.xy;

  for (int i=0; i<numOctave; ++i) {
    sum += w * pnoise(f*v, f*scaledTileRes);
    
    f *= frequency;
    w *= amplitude;
  }

  return sum;
}

// Derivative Perlin Noise fbm 2D
float fbm_dpnoise(in vec2 pt,
                  in float zoom,
                  const int numOctave, 
                  const float frequency, 
                  const float amplitude)
{
  float sum = 0.0f;

  float f = frequency;
  float w = amplitude;
  vec2 dn = vec2(0.0f);

  vec2 v = zoom * pt;
  vec2 scaledTileRes = zoom * uTileRes.xy;

  for (int i=0; i<numOctave; ++i) {
    vec3 n = dpnoise(f*v, f*scaledTileRes);
    dn += n.yz;
    
    float crestFactor = 1.0f / (1.0f + dot(dn,dn));
    
    sum += w * n.x * crestFactor;
    
    f *= frequency;
    w *= amplitude;
  }

  return sum;
}




//---------------------------- NOISE 3D


// Classical Perlin Noise 3D
float pnoise(vec3 pt, vec3 scaledTileRes = vec3(0.0f)) {
  // Retrieve the integral part (for indexation)
  vec3 ipt0 = floor(pt);
  vec3 ipt1 = ipt0 + vec3(1.0f);

  // Tile the noise (if enabled)
  if (uEnableTiling)
  {
    ipt0 = mod(ipt0, scaledTileRes);
    ipt1 = mod(ipt1, scaledTileRes);
  }
  ipt0 = mod289(ipt0);
  ipt1 = mod289(ipt1);

  // Compute the 8 corners hashed gradient indices
  vec4 ix = vec4(ipt0.x, ipt1.x, ipt0.x, ipt1.x);
  vec4 iy = vec4(ipt0.yy, ipt1.yy);
  vec4 p = permute(permute(ix) + iy);
  vec4 p0 = permute(p + ipt0.zzzz);
  vec4 p1 = permute(p + ipt1.zzzz);

  // Compute Pseudo Random Numbers
  vec4 gx0 = p0 * (1.0f / 7.0f);
  vec4 gy0 = fract(floor(gx0) * (1.0f / 7.0f)) - 0.5f;
  gx0 = fract(gx0);  
  vec4 gz0 = vec4(0.5f) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0.0f));
  gx0 -= sz0 * (step(0.0f, gx0) - 0.5f);
  gy0 -= sz0 * (step(0.0f, gy0) - 0.5f);

  vec4 gx1 = p1 * (1.0f / 7.0f);
  vec4 gy1 = fract(floor(gx1) * (1.0f / 7.0f)) - 0.5f;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(0.5f) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0.0f));
  gx1 -= sz1 * (step(0.0f, gx1) - 0.5f);
  gy1 -= sz1 * (step(0.0f, gy1) - 0.5f);
  /*------------------------*/

  // Create unnormalized gradients
  vec3 g000 = vec3(gx0.x, gy0.x, gz0.x);
  vec3 g100 = vec3(gx0.y, gy0.y, gz0.y);
  vec3 g010 = vec3(gx0.z, gy0.z, gz0.z);
  vec3 g110 = vec3(gx0.w, gy0.w, gz0.w);
  vec3 g001 = vec3(gx1.x, gy1.x, gz1.x);
  vec3 g101 = vec3(gx1.y, gy1.y, gz1.y);
  vec3 g011 = vec3(gx1.z, gy1.z, gz1.z);
  vec3 g111 = vec3(gx1.w, gy1.w, gz1.w);

  // 'Fast' normalization
  vec4 dp = vec4(dot(g000, g000), dot(g100, g100), dot(g010, g010), dot(g110, g110));
  vec4 norm = inversesqrt(dp);
  g000 *= norm.x;
  g100 *= norm.y;
  g010 *= norm.z;
  g110 *= norm.w;

  dp = vec4(dot(g001, g001), dot(g101, g101), dot(g011, g011), dot(g111, g111));
  norm = inversesqrt(dp);
  g001 *= norm.x;
  g101 *= norm.y;
  g011 *= norm.z;
  g111 *= norm.w;

  // Retrieve the fractional part (for interpolation)
  vec3 fpt0 = fract(pt);
  vec3 fpt1 = fpt0 - vec3(1.0f);

  // Calculate gradient's influence
  float n000 = dot(g000, fpt0);
  float n100 = dot(g100, vec3(fpt1.x, fpt0.yz));
  float n010 = dot(g010, vec3(fpt0.x, fpt1.y, fpt0.z));
  float n110 = dot(g110, vec3(fpt1.xy, fpt0.z));
  float n001 = dot(g001, vec3(fpt0.xy, fpt1.z));
  float n101 = dot(g101, vec3(fpt1.x, fpt0.y, fpt1.z));
  float n011 = dot(g011, vec3(fpt0.x, fpt1.yz));
  float n111 = dot(g111, fpt1);

  // Interpolate gradients
  vec3 u = fade(fpt0);
  float nxy0 = mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y);
  float nxy1 = mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y);
  float noise = mix(nxy0, nxy1, u.z);

  return noise;
}


// Classical Perlin Noise 2D + time
float pnoiseLoop(vec2 u, float dt) {
  vec3 pt1 = vec3(u, dt);
  vec3 pt2 = vec3(u, dt-1.0f);

  return mix(pnoise(pt1), pnoise(pt2), dt);
}


// Derivative Perlin Noise 3D
//vec4 dpnoise(vec3 pt);



// -- Fractional Brownian Motion function --

float fbm_pnoise(vec3 pt, 
                 float zoom,
                 const int numOctave, 
                 const float frequency, 
                 const float amplitude)
{
  float sum = 0.0f;
  float f = frequency;
  float w = amplitude;

  vec3 v = zoom * pt;
  vec3 scaledTileRes = zoom * uTileRes;

  for (int i=0; i<numOctave; ++i) {
    sum += w * pnoise(f*v, f*scaledTileRes);

    f *= frequency;
    w *= amplitude;
  }
  
  return sum;
}

//float fbm_dpnoise(vec3 pt, float zoom);

