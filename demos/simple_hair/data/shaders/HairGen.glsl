/*
 *          HairGen.glsl
 *
 *
 *
 */

//------------------------------------------------------------------------------


-- VS


layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inTangent;

out VDataBlock {
  vec3 position;
  vec3 tangent;
} OUT;

void main()
{
  OUT.position = inPosition;
  OUT.tangent = inTangent;
}


--

//------------------------------------------------------------------------------

-- TCS

uniform int uNumLines           = 1;
uniform int uNumLinesSubSegment = 1;

/// Inputs
in VDataBlock {
  vec3 position;
  vec3 tangent;
} IN[];

/// Outputs
layout(vertices = 4) out;

out VDataBlock {
  vec3 position;
  vec3 tangent;
} OUT[];


void main() {
# define ID  gl_InvocationID

  OUT[ID].position = IN[ID].position;
  OUT[ID].tangent  = IN[ID].tangent;

  if (ID == 0) {
    gl_TessLevelOuter[0] = uNumLines;
    gl_TessLevelOuter[1] = uNumLinesSubSegment;
  }
}


--

//------------------------------------------------------------------------------

-- TES


uniform mat4 uMVP;

layout(isolines) in;
//layout(point_mode) in;

in VDataBlock {
  vec3 position;
  vec3 tangent;
} IN[];


out VDataBlock {
  vec4 position;
} OUT;

vec4 hermit_mix(in vec3 p0, in vec3 p1, in vec3 t0, in vec3 t1, in float u);

void main() {
  /// for each line strip :
  /// gl_TessCoord.x is in range [0, 1[
  /// gl_TessCoord.y is constant
  vec2 uv = gl_TessCoord.xy;

  vec3 p0 = mix(IN[0].position, IN[1].position, uv.y);
  vec3 p1 = mix(IN[2].position, IN[3].position, uv.y);
  vec3 t0 = mix(IN[0].tangent, IN[1].tangent, uv.y);
  vec3 t1 = mix(IN[2].tangent, IN[3].tangent, uv.y);

  vec4 lerpedPos = hermit_mix(p0, p1, t0, t1, uv.x);

  gl_Position = uMVP * lerpedPos;
}


const mat4 mHermit = mat4(
  vec4( 2.0f, -3.0f, 0.0f, 1.0f),
  vec4(-2.0f,  3.0f, 0.0f, 0.0f),
  vec4( 1.0f, -2.0f, 1.0f, 0.0f),
  vec4( 1.0f, -1.0f, 0.0f, 0.0f)
);

vec4 hermit_mix(in vec3 p0, in vec3 p1, in vec3 t0, in vec3 t1, in float u) {
  vec4 vU = vec4(u*u*u, u*u, u, 1.0f);

  mat4 B = mat4( 
    vec4(p0.x, p1.x, t0.x, t1.x),
    vec4(p0.y, p1.y, t0.y, t1.y),
    vec4(p0.z, p1.z, t0.z, t1.z),
    vec4(1.0f, 1.0f, 0.0, 0.0f)
  );

  return vU * mHermit * B;
}


--


//------------------------------------------------------------------------------

-- FS


layout(location = 0) out vec4 fragColor;

void main() {
  vec3 color = vec3(1.0f,0.0f,0.0f);
  fragColor = vec4(color, 1.0f);
}
