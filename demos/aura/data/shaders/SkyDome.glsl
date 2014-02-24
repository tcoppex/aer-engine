// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------
//        Procedural sky
//------------------------------------------------------------------------------


-- VS

// IN
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;

// OUT
out VDataBlock {
  vec2  texcoord1;
  vec2  texcoord2;
  float intensity;
  float intensity_sqr;
  vec3  position;
} OUT;

// UNIFORM
uniform mat4 uModelViewProjMatrix;
uniform float uSkyClock;


void main() {
  OUT.texcoord1  = inTexCoord + uSkyClock * vec2(0.33f, 0.66f);
  OUT.texcoord1 *= 2.5f;
  
  OUT.texcoord2  = inTexCoord + uSkyClock * vec2(1.33f, 1.66f);
  OUT.texcoord2 *= 3.5f;
  
  // This can be precomputed  
  OUT.intensity     = mix(0.370f, 0.610f, inPosition.y);
  OUT.intensity_sqr = OUT.intensity * OUT.intensity;
  
  // decrease level to hide the limit
  OUT.position    = inPosition;
  OUT.position.y -= 0.45f;
  
  gl_Position  = uModelViewProjMatrix * vec4(OUT.position, 1.0f);
}

--

//------------------------------------------------------------------------------


-- FS


// IN
in VDataBlock {
  vec2  texcoord1;
  vec2  texcoord2;
  float intensity;
  float intensity_sqr;
  vec3  position;
} IN;

// OUT
layout(location = 0) out vec4 fragColor;

// UNIFORM
uniform sampler2D uSkyTex;
uniform vec3 uSkyColor;


void main() {  
  vec3 sky1 = texture(uSkyTex, IN.texcoord1).rgb;
  vec3 sky2 = texture(uSkyTex, IN.texcoord2).rgb;

  vec3 cloud1 = sky1 + sky2;
  vec3 cloud2 = sky1 * sky2;


  // Smooth out the effect (on border)
  float shadeOut = (1.0f - IN.position.y) * dot(IN.position, IN.position);
        shadeOut = smoothstep(0.0f, 1.0f, shadeOut);

  vec3 cloudColor = mix(cloud1, cloud2, shadeOut);
       cloudColor *= IN.intensity_sqr;
  
  vec3 skyColor = uSkyColor;
       skyColor.rg *= (1.0f - IN.intensity);
       skyColor.b  *= IN.intensity;

  fragColor.rgb = skyColor * (1.0f - cloudColor.r) + cloudColor;

  // In the deferred pipeline, an alpha of 0 means the light will not be computed
  // for it.
  fragColor.a = 0.0f;
}

--


//------------------------------------------------------------------------------


-- RenderTexture.FS

/*
  Fragment program used to create the cloud texture
  (needs to include Noise.glsl as a fragment shader too).
  
  PostProcess.Vertex is used as vertex shader.
*/

// Noise prototypes
float normalizeNoise(float n);
float fbm_pnoise(vec2, float, const int, const float, const float);
float fbm_dpnoise(vec2, float, const int, const float, const float);
float fbm_pnoise(vec3, float, const int, const float, const float);


// IN
in VDataBlock {
  vec2 texCoord;
} IN;

// OUT
layout(location=0) out vec4 fragColor;

vec4 computeCloudTexture(vec2 v) {
  const int octave = 4;
  const float zoom = 1.0f / 256.0f; // to tile, must be a power of two
  const float freq = 2.0f;          // to tile, must be an integer
  const float w    = 0.45f;

  // Just a really basic noise without post processing
  float noise = 1.5f * fbm_pnoise(v, zoom, octave, freq, w);

  return vec4(vec3(noise), 1.0f);
}

void main() {
  fragColor = computeCloudTexture(gl_FragCoord.xy);
}

