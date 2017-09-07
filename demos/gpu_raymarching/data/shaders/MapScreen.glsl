/*
 *          MapScreen.glsl
 *
 */

//------------------------------------------------------------------------------


-- VS


out VDataBlock {
  vec2 texCoord;
} OUT;

void main() {
  OUT.texCoord.s = (gl_VertexID << 1) & 2;
  OUT.texCoord.t = gl_VertexID & 2;

  gl_Position = vec4(2.0f * OUT.texCoord - 1.0f, 0.0f, 1.0f);
}


--

//------------------------------------------------------------------------------


-- FS

in VDataBlock {
  vec2 texCoord;
} IN;

layout(location = 0) out vec4 fragColor;

uniform float uTime;
uniform vec2  uResolution;

//-------------------


// CSG operations
#define UNION(a, b)         ((a.x<b.x) ? a : b )
#define INTERSECTION(a, b)  ((a.x<b.x) ? b : a )
#define SUBSTRACTION(a, b)  ((-(a.x)<b.x) ? b : a )


// PRIMITIVEs
float sdCube(vec3 p, vec3 b) {
  vec3 d = abs(p) - b;
  float minDist = min(max(d.x, max(d.y, d.z)), 0.0);
  return minDist + length(max(d, 0.0));
}

float sdSphere(vec3 p, float s) {
  return length(p) - s;
}

float sdCylinder(vec3 p, vec3 c) {
  return max(length(p.xz)-c.z, abs(p.y)-c.y);
}

float udRoundBox(vec3 p, vec3 b, float r) {
  return length(max(abs(p)-b,0.0))-r;
}

//-------------------

/// Render FUTURAMA's Bender burglar face via raymarching

#define MAT_METAL     0.0f
#define MAT_EYE       1.0f
#define MAT_MOUTH     2.0f
#define MAT_BLACK     3.0f


// Test a point against all objects in the scene
float scene(vec3 p, out float m) {
  vec2 final;
  
  // head
  vec2 obj1;
  obj1.x = min(sdCylinder(p, vec3(0.0f, 1.2f, 0.9f)), sdSphere(p-vec3(0.0f,1.2f,0.0f), 0.9f));
  obj1.y = MAT_METAL;
    
  // visiere  
  vec2 obj2 = vec2(udRoundBox(p-vec3(0.0f,0.3f,0.0f), vec3(0.7f,0.18f,1.2f), 0.23f), MAT_METAL);
  float tmp_x = obj2.x;
  obj2.x = max(-sdCube(p-vec3(0.0f,0.0f,1.6f), vec3(1.0f, 1.0f, 0.35f)), obj2.x);
  float opener = smoothstep(0.2f,1.0f,0.5+0.5f*sin(uTime));
  obj2.x = max(-udRoundBox(p-vec3(0.0f,0.3f,1.5f), vec3(0.7f,0.18f*opener,0.4f), 0.2f), obj2.x);

  obj2.y = (obj2.x == tmp_x) ? obj2.y : MAT_BLACK;
  
  
  // eyeballs
  vec2 obj3;  
  vec3 eye = vec3(0.45, 0.25, 0.8);
  obj3.x = sdSphere(p-eye, 0.42);
  eye.x = -eye.x;
  obj3.x = min(obj3.x, sdSphere(p-eye, 0.42));
  obj3.y = MAT_EYE;
  
  // cavity for the mouth
  vec2 obj4 = vec2(udRoundBox(p-vec3(0.0f, -0.65, 0.7), vec3(0.6, 0.07, 0.2), 0.2), MAT_METAL); 
  
  //mouth
  vec2 obj5 = vec2(sdCylinder(p-vec3(0.0f,-0.6,0.0), vec3(0.0f, 0.5, 0.85)), MAT_MOUTH);

  
  // CSG operations
  final = obj1;
  final = UNION(final, obj2);
  final = UNION(final, obj3);
  final.x = max(-obj4.x, final.x);
  final = UNION(final, obj5);


  // --
  float d;
  d = final.x;
  m = final.y;

  return d;
}

vec3 scene_normal(vec3 pos) {
  vec2 eps = vec2(0.001, 0.0);
  float dummy_m;
  vec3 n;
  
  float d = scene(pos, dummy_m);
  n.x = scene(pos + eps.xyy, dummy_m) - d;
  n.y = scene(pos + eps.yxy, dummy_m) - d;
  n.z = scene(pos + eps.yyx, dummy_m) - d;
  return normalize(n);
}

vec3 trace(vec3 ro, vec3 rd, out float m, out bool hit) {
  const int   kMaxSteps = 64;  
  const float kHitThreshold = 0.001;

  vec3 pos = ro;
  hit = false;

  for (int i=0; i<kMaxSteps; ++i) {
    float d = scene(pos, m);

    if (d<kHitThreshold) {
      hit = true;
      break;
    }

    pos += d * rd;
  }
 
  return pos;
}


// Color the fragment depending on the material index
vec3 shade(vec3 pos, vec3 n, float m) {
  vec3 col = vec3(0.2f);

  if (m == MAT_METAL) {
    col = vec3(0.5f, 0.52f, 0.55f);
  } else if (m == MAT_MOUTH) {   
    col = vec3(1.0f, 1.0f, 0.9f);
    col *= smoothstep(0.0f, 1.0f, mod(abs(8.0f*(pos.y)), 2.0f));
    col *= smoothstep(0.0f, 1.0f, mod(abs(15.0f*pos.x+100.0f), 4.0f));
  } else if (m == MAT_EYE) {
    col = vec3(1.0f, 1.0f, 0.8f);

    vec3 look = normalize(vec3(sin(1.5f*uTime)*abs(pos.x), 0.0f, 2.0f));
    col -= smoothstep(0.0f, 1.0f, pow(dot(look, n), 40.0f)); 
  } else if (m == MAT_BLACK) {
    col = vec3(0.1f, 0.1f, 0.1f);
  } else {
  }

  col *= 0.5f*(n + 1.0f);

  float rTheta = 0.8f * (0.5f+0.5f*sin(uTime));
  vec2 cs = vec2(cos(rTheta), sin(rTheta));
  mat3 rotY = mat3(vec3( cs.x, 0.0f, cs.y), 
            vec3(  0.0f, 1.0f, 0.0f),
            vec3(-cs.y, 0.0f, cs.x) );
  vec3 lightDir = normalize(vec3(2.0f, -2.0f, -5.0f));
  lightDir = lightDir * rotY;
  float diff = dot(n, -lightDir);

  return diff * col;
}

void main() {
  vec2 uv = gl_FragCoord.xy / uResolution.xy;
  vec2 p = -1.0f + 2.0f * uv;                       // pixel coord in [-1, 1]
  p.x *= uResolution.x / uResolution.y;             // change aspect ratio for a quad

  // Default camera
  vec3 ro = vec3(0.0f, 1.0f, 5.0f);                 // RayOrigin, or "cameraEye"
  vec3 rd = normalize(vec3(p.x, p.y, -1.0f));       // RayDirectrion, or "cameraDirection"

  // Update camera [optional]
  //..

  float m = 0.0f;                                   // material hit index  
  bool hit;
  vec3 pos = trace(ro, rd, m, hit);                 // Trace a ray
  vec3 col = vec3(0.2f);                            // Default background color

  if (hit) {
    vec3 n = scene_normal(pos);
    col = shade(pos, n,  m);
  }

  fragColor = vec4(col, 1.0f);
}