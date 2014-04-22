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
  int instanceID;
} OUT;

void main() {
  OUT.position = inPosition;
  OUT.tangent  = inTangent;

  OUT.instanceID = gl_InstanceID;
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

  int instanceID;
} IN[];

/// Outputs
layout(vertices = 6) out;

out VDataBlock {
  vec3 position;
  vec3 tangent;
} OUT[];

patch out int tc_instanceID;

float calculate_tri_area(in vec3 A, in vec3 B, in vec3 C);

void main() {
# define ID  gl_InvocationID

  OUT[ID].position = IN[ID].position;
  OUT[ID].tangent  = IN[ID].tangent;

  if (ID == 0) {
    // Tesselation params
    gl_TessLevelOuter[0] = uNumLines;
    gl_TessLevelOuter[1] = uNumLinesSubSegment;

    tc_instanceID  = IN[0].instanceID;
  }
}


float calculate_tri_area(in vec3 A, in vec3 B, in vec3 C) {
/// Calculate triangle area
    vec3 AB = B - A;
    vec3 AC = C - A;

    // projection of C on AB
    float kAP = dot(AC, AB);

    // height (CP) length
    float sqr_AP = kAP;
    float sqr_AC = dot(AC, AC);
    float kCP = sqrt(sqr_AC - sqr_AP);

    // base (AB) length
    float kAB = sqrt(dot(AB, AB));

    // Triangle area
    float area = 0.5f * kAB * kCP;

    return area;
}


--

//------------------------------------------------------------------------------

-- TES


/// ### uniforms ###
uniform mat4 uMVP;
uniform sampler1D uTexRandom;
uniform int uNumInstances = 1;

/// ### inputs ###
layout(isolines) in;
//layout(point_mode) in;

in VDataBlock {
  vec3 position;
  vec3 tangent;
} IN[];

patch in int tc_instanceID;

/// ### outputs ###
out float te_colorFactor;

/// ### functions ###
vec4 hermit_mix(in vec3 p0, in vec3 p1, in vec3 t0, in vec3 t1, in float u);

void main() {
  /// for each line strip :
  /// gl_TessCoord.x is in range [0, 1[
  /// gl_TessCoord.y is constant
  const vec2 uv = gl_TessCoord.xy;


  /// Calculate the interpolate point at each vertex
  vec4 p0 = hermit_mix(IN[0].position, IN[1].position,
                       IN[0].tangent,  IN[1].tangent,
                       uv.x);
  vec4 p1 = hermit_mix(IN[2].position, IN[3].position,
                       IN[2].tangent,  IN[3].tangent,
                       uv.x);
  vec4 p2 = hermit_mix(IN[4].position, IN[5].position,
                       IN[4].tangent,  IN[5].tangent,
                       uv.x);

  /// Calculate random texture coordinates
  const float texelSize = 1.0f / (textureSize(uTexRandom, 0).x-1.0f);
  float texcoord = uv.y + tc_instanceID * texelSize;

  /// Retrieve random coordinates
  vec3 rand_coords = texture(uTexRandom, texcoord).rgb;

  /// Calculate the final lerped position
  vec4 lerped_pos = rand_coords.x*p0 + rand_coords.y*p1 + rand_coords.z*p2;

  /// Set OUTPUTS
  gl_Position    = uMVP * lerped_pos;
  te_colorFactor = float(tc_instanceID + 1.0f) / float(uNumInstances);
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


in float te_colorFactor;

layout(location = 0) out vec4 fragColor;

void main() {
  vec3 color = te_colorFactor * vec3(0.3f, 0.2f, 0.0f);
  fragColor = vec4(color, 1.0f);
}
