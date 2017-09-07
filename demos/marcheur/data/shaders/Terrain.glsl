// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
//------------------------------------------------------------------------------


-- VS

//----------------------------------------
// Noise functions prototypes
//----------------------------------------
//#include "Noise.Include"
float normalizeNoise(float n);
float fbm_pnoise(vec2, float, const int, const float, const float);
float fbm_dpnoise(vec2, float, const int, const float, const float);
float fbm_pnoise(vec3, float, const int, const float, const float);

//----------------------------------------

uniform mat4 uModelViewProjMatrix;
uniform float uTime = 0.0f;

in layout(location = 0) vec4 inPosition;
in layout(location = 1) vec3 inNormal;
in layout(location = 2) vec2 inTexCoord;

out VDataBlock {
  vec3 normalVS;
  vec2 texCoord;
} OUT;


void main() {
  const int octave = 8;
  const float zoom = 1.0f / 256.0f;
  const float freq = 4.0f;
  const float w    = 0.745f;

  vec2 v = inPosition.xz + vec2(0.0f, 0.01f*uTime);
  float noise = 1.5f * fbm_pnoise(v, zoom, octave, freq, w);

  vec4 position = inPosition;
  position.y += noise * float(abs(inPosition.x) > 2.5f);
  gl_Position = uModelViewProjMatrix * position;

  vec3 normal = inNormal + 0.4f*vec3(noise, 2.0f, noise);
  OUT.normalVS = normalize(normal); // uNormalViewMatrix * inNormal

  OUT.texCoord = inTexCoord;
}


--

//------------------------------------------------------------------------------


-- FS


uniform vec3 uColor = vec3(1.0f);
uniform bool uEnableTexturing = false;
uniform sampler2D uTexture;


in VDataBlock {
  vec3 normalVS;
  vec2 texCoord;
} IN;

out layout(location = 0) vec4 outDiffuse;
out layout(location = 1) vec3 outNormal;


void main()
{
  // == Diffuse color ==
  vec4 fragDiffuse = vec4(uColor, 1.0f);   
  if (uEnableTexturing) {
    fragDiffuse = texture(uTexture, IN.texCoord);
  }

  // == Normal [View-Space] ==
  // [ TODO: Store only the XY component and retrieve the Z later ]
  // works when CullFace enables (to avoid z-fighting jaggies)
  vec3 fragNormal = (gl_FrontFacing) ? IN.normalVS : -IN.normalVS; 
  fragNormal = normalize(fragNormal);

  fragDiffuse *= dot(fragNormal, normalize(vec3(2.f, 5.0f, 0.0f)));

  // Output  
  outDiffuse = fragDiffuse;
  outNormal  = fragNormal;

  //outDiffuse.rgb = 0.5f*fragNormal + 0.5f;
}
