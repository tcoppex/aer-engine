// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------
// - Variance Shadow Mapping -
//
// Original algorithm from William Donnelly and Andrew Lauritzen
// reference : http://http.developer.nvidia.com/GPUGems3/gpugems3_ch08.html
// -----------------------------------------------------------------------------

-- create.VS

/// Simple passthrough to calculate the VSM shadow map

layout(location=0) in vec4 inPosition;

out VDataBlock {
  vec4 position;
} OUT;

uniform mat4 uViewProjectionMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uModelMatrix;

void main() {
  gl_Position = (uViewProjectionMatrix * uModelMatrix) * inPosition;

//------------
  OUT.position = (uViewMatrix * uModelMatrix) * inPosition;
//------------
}


--

//------------------------------------------------------------------------------

-- create.FS

in VDataBlock {
  vec4 position;
} IN;

out vec2 outVarianceShadowMap;

void main() {
  float depth = 0.0f;

//------------
  depth = -IN.position.z;
//------------

  outVarianceShadowMap = vec2(depth, depth*depth);

  // Bias calculation
  float ddx = dFdx(depth);
  float ddy = dFdy(depth);
  outVarianceShadowMap.y += 0.25f * (ddx*ddx + ddy*ddy);
}

--

//------------------------------------------------------------------------------

-- render.include

// hacking the variance works better than the original algorithm for me
// & i cant figure out why
uniform float uMinVariance = 99.0f;

float calculate_vsm_shadow(in sampler2D tex, in vec3 coords) {
  vec2 moments = texture(tex, coords.xy).rg;

  /// Chebyshev upper bound
  float variance = moments.y - moments.x*moments.x;
        variance = max(variance, uMinVariance);  
  float d = coords.z - moments.x;
  float p_max = variance / (variance + d*d);
  float p = float(d <= 0.0f);

  return max(p, p_max);
}

--

//------------------------------------------------------------------------------

-- render.VS

uniform mat4 uViewProjectionMatrix;
uniform mat4 uModelMatrix;
uniform mat4 uShadowMatrix;
uniform mat4 uShadowProjMatrix;

in layout(location = 0) vec4 inPosition;

out VDataBlock {
  vec3 coords;
} OUT;


void main() {
  gl_Position = (uViewProjectionMatrix * uModelMatrix) * inPosition;

//------------
  vec4 shadowView     = (uShadowMatrix * uModelMatrix) * inPosition;
  vec4 shadowViewProj = (uShadowProjMatrix * uModelMatrix) * inPosition;

  OUT.coords.xy = 0.5f*(shadowViewProj.xy / shadowViewProj.w) + 0.5f;
  OUT.coords.z  = -shadowView.z;
//------------
}


--

//------------------------------------------------------------------------------

-- render.FS

uniform vec3 uColor = vec3(1.0f);
uniform sampler2D uShadowMap;

float calculate_vsm_shadow(in sampler2D tex, in vec3 coords);


in VDataBlock {
  vec3 coords;
} IN;

out layout(location = 0) vec4 outDiffuse;

void main() {
  float shadow = 1.0f;

//------------
  vec3 coords = IN.coords;
  shadow = calculate_vsm_shadow(uShadowMap, coords);
//------------

  shadow = clamp(shadow, 0.15f, 1.0f);
  outDiffuse.rgb = uColor * shadow;
}
