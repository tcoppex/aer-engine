/*
 *          Density.glsl
 *
 *      CSG operations, primitives functions and main density function.
 *      
 *      ref:
 *       -"Modeling with distance functions" by Inigo Quilez
 *        http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
 *
 *
 *    This code must be include manually in the program shader using the
 *    compute_density function.
 */

//------------------------------------------------------------------------------


-- Include


//----------------------------------------
// Noise functions prototypes
//----------------------------------------
//#include "Noise.Include"
float normalizeNoise(float n);
float fbm_pnoise(vec2, float, const int, const float, const float);
float fbm_dpnoise(vec2, float, const int, const float, const float);
float fbm_pnoise(vec3, float, const int, const float, const float);

//--------------
mat3 m = mat3(0.00,  0.80,  0.60,
              -0.80,  0.36, -0.48,
              -0.60, -0.48,  0.64);

float hash(float n) {
  return fract(sin(n)*43758.5453);
}

float noise(in vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*57.0 + 113.0*p.z;

    float res = mix(mix(mix(hash(n+  0.0), hash(n+  1.0),f.x),
                        mix(hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
                    mix(mix(hash(n+113.0), hash(n+114.0),f.x),
                        mix(hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
    return res;
}

float fbm(vec3 p) {
    float f = 0.0;

    f += 0.5000*noise(p); p = m*p*2.02;
    f += 0.2500*noise(p); p = m*p*2.03;
    f += 0.1250*noise(p); p = m*p*2.01;
    f += 0.0625*noise(p);

    return f/0.9375;
}
//--------------

float fbm_3d(in vec3 ws) {
  const float N = 128.f;
  const float zoom = 1/N;       // to tile, must be a power of two
  const int octave = 8;
  const float freq = 2.0f;      // to tile, must be an integer
  const float w    = 0.45f;

  return N * fbm_pnoise(ws, zoom, octave, freq, w);
}



//----------------------------------------
// CSG operations
//----------------------------------------
float opUnion(float d1, float d2) {
  return min(d1, d2);
}

float opSmoothUnion(float d1, float d2, float k) {
  float r = exp(-k*d1) + exp(-k*d2);
  return -log(r) / k;
}

float opIntersection(float d1, float d2) {
  return max(d1, d2);
}

float opSubstraction(float d1, float d2) {
  return max(d1, -d2);
}

vec3 opRepeat(vec3 p, vec3 c) {
  return mod(p, c) - 0.5f*c;
}

float opDisplacement(vec3 p, float d) {
  p = d*p;
  return sin(p.x)*sin(p.y)*sin(p.z);
}

//----------------------------------------
// PRIMITIVEs
//----------------------------------------
/*
float sdCube(vec3 p, vec3 b) {
  vec3 d = abs(p) - b;
  float minDist = min(max( d.x, max(d.y, d.z)), 0.0f);
  return minDist + 0*length(max(d, 0.0f));
}*/


float sdPlane(vec3 p, vec4 n) {
  //n.xyz = normalize(n.xyz);
  return n.w + dot(p, n.xyz);
}

float sdSphere(vec3 p, float s) {
  return length(p) - s;
}

float udRoundBox(vec3 p, vec3 b, float r) {
  return length(max(abs(p)-b, 0.0f)) - r;
}

float sdCylinder(vec3 p, float c) {
  //return opIntersection(length(p.xz-c.xy) - c.z, abs(p.y)-c.y);
  return length(p.xy) - c;
}

float sdTorus(vec3 p, vec2 t) {
  vec2 q = vec2(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}

mat3 rotX(float a) {
  float c = cos(a);
  float s = sin(a);

  return
  mat3(1.0f, 0.0f, 0.0f,
       0.0f,    c,   -s,
       0.0f,    s,    c);

}


float terrain(in vec3 ws) {
  float d = 0.0f;

  //d = sdPlane(ws, vec4(0.0f, 1.0f, 0.0f, 0.0f));

  float N = 5.0f;

  float noise = fbm_3d(ws*N);
  d += 2.f*fbm_3d(ws*N) / N;

  ws += 0.05f*noise; 
  float radius = 20.0f;
  d = opIntersection(d, sdSphere(ws, radius));

  float d2 = sdTorus(ws + 0.25f*fbm_3d(ws), vec2(1.2f*radius, 1.0f));
  d = opSmoothUnion(d, d2, 0.75f);

  return d;
}

//----------------------------------------
// Main density function
//----------------------------------------
float compute_density(in vec3 ws) {
  float d = 0.0f;

  return terrain(ws);

  //ws = ws * rotX(radians(90.0f));

  //d += sdPlane(ws, vec4(0.0f, 1.0f, 0.0f, 0.0f));
  //d += sdCylinder(ws-vec3(0.0f), 1.0f);
  d += sdTorus(ws + 0.25f*fbm_3d(ws), vec2(20.0f, 1.0f));

  ws *= 1.0f / 2.5f;

  float dx = 10.5f;
  vec3 ws2 = ws + fbm_3d(dx*ws) / 55.f;

  ws2 += opDisplacement(ws, 0.7f);
  //ws2 = opRepeat(ws, vec3(16.0f + 0.2f*fbm_3d(ws)));

  float d2 = 0.0f;
  d2 = sdSphere(ws2 + vec3(0.0f,0.f,+7.0f), 7.0f);

  //d = opUnion(d, d2);
  d = opSmoothUnion(d, d2, 0.75f);


  return d;
}
