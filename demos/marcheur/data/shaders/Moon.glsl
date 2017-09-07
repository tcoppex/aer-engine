// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
//------------------------------------------------------------------------------


-- VS

uniform mat4 uModelViewProjMatrix;
uniform float uTime = 0.0f;

in layout(location = 0) vec4 inPosition;


void main() {
  gl_Position = uModelViewProjMatrix * inPosition;
}


--

//------------------------------------------------------------------------------


-- FS


//----------------------------------------
// Noise functions prototypes
//----------------------------------------
//#include "Noise.Include"
float normalizeNoise(float n);
float fbm_pnoise(vec2, float, const int, const float, const float);
float fbm_dpnoise(vec2, float, const int, const float, const float);
float fbm_pnoise(vec3, float, const int, const float, const float);

//----------------------------------------

uniform vec3 uColor = vec3(1.0f);

out layout(location = 0) vec4 outDiffuse;

void main()
{
  // == Diffuse color ==
  vec4 fragDiffuse = vec4(uColor, 1.0f);

  // Output  
  outDiffuse = fragDiffuse;
}
