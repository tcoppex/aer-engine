// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------
//        Skinning + BlendShape on GPU
//------------------------------------------------------------------------------


-- VS


#define NO_JOINT      0xff


// Transformation
uniform mat4 uModelViewProjMatrix;
uniform mat3 uNormalMatrix = mat3(1.0f); //

// Skinning
layout(binding=0) uniform samplerBuffer uSkinningDatas;
//uniform isamplerBuffer uSkinningIndices;
//uniform  samplerBuffer uSkinningWeights;

// Blend Shape
                  uniform           uint uBS_count;   ///< number of blendshape in data
                  uniform           uint uBS_used;    ///< number of used blendshape
layout(binding=1) uniform  samplerBuffer uBS_weights; ///< user specified weights
layout(binding=2) uniform isamplerBuffer uBS_indices; ///< user specified indices
layout(binding=3) uniform  samplerBuffer uBS_data;    ///< mesh's blendshapes
layout(binding=4) uniform isamplerBuffer uBS_LUT;     ///< VertexID to blendshape ID LUT

// Input attributes
layout(location = 0) in  vec3 inPosition;
layout(location = 1) in  vec3 inNormal;
layout(location = 2) in  vec2 inTexCoord;
layout(location = 3) in ivec4 inJointIndices; //
layout(location = 4) in  vec3 inJointWeight;  //

// Output attributes
out VDataBlock {
  vec3 normal;
  vec2 texCoord;
} OUT;



///--------------------------------------------------------------------
/// SKINNING
///--------------------------------------------------------------------
subroutine void skinning_subroutine(in vec4 weights, inout vec3 v, inout vec3 n);
subroutine uniform skinning_subroutine uSkinning;

void calculate_skinning(inout vec3 position, inout vec3 normal) {
  if (!(inJointWeight.x > 0.0f)) {
    return;
  }

  vec4 weights   = vec4(inJointWeight.xyz, 0.0f);
       weights.w = 1.0f - (weights.x + weights.y + weights.z);

  uSkinning(weights, position, normal);
}

///--------------------------------------------------------------------
/// SKINNING : DUAL QUATERNION BLENDING
///--------------------------------------------------------------------

void getDualQuaternions(out mat4 Ma, out mat4 Mb) {
  ivec4 indices = 2 * inJointIndices;

  /// Retrieve the real (Ma) and dual (Mb) part of the dual-quaternions
  Ma[0] = texelFetch(uSkinningDatas, indices.x+0);
  Mb[0] = texelFetch(uSkinningDatas, indices.x+1);

  Ma[1] = texelFetch(uSkinningDatas, indices.y+0);
  Mb[1] = texelFetch(uSkinningDatas, indices.y+1);

  Ma[2] = texelFetch(uSkinningDatas, indices.z+0);
  Mb[2] = texelFetch(uSkinningDatas, indices.z+1);

  Ma[3] = texelFetch(uSkinningDatas, indices.w+0);
  Mb[3] = texelFetch(uSkinningDatas, indices.w+1);
}

subroutine(skinning_subroutine)
void skinning_DQBS(in vec4 weights, inout vec3 v, inout vec3 n) {
  /// Paper :
  ///   "Geometric Skinning with Approximate Dual Quaternion Blending"
  ///   - Kavan et al 2008

  // Retrieve the dual quaternions
  mat4 Ma, Mb;
  getDualQuaternions(Ma, Mb);

  // Handles antipodality by sticking joints in the same neighbourhood
  weights.xyz *= sign(Ma[3] * mat3x4(Ma));

  // Apply weights
  vec4 A = Ma * weights;  // real part
  vec4 B = Mb * weights;  // dual part

  // Normalize
  float invNormA = 1.0f / length(A);
  A *= invNormA;
  B *= invNormA;

  // Position
  v += 2.0f * cross(A.xyz, cross(A.xyz, v) + A.w*v);              // rotate
  v += 2.0f * (A.w * B.xyz - B.w * A.xyz + cross(A.xyz, B.xyz));  // translate

  // Normal
  n += 2.0f * cross(A.xyz, cross(A.xyz, n) + A.w*n);
}


///--------------------------------------------------------------------
/// SKINNING : LINEAR BLENDING
///--------------------------------------------------------------------

void getSkinningMatrix(in int jointId, out mat3x4 skMatrix) {
  if (jointId == NO_JOINT) {
    skMatrix = mat3x4(1.0f);
    return;
  }

  const int matrixId = 3*jointId;
  skMatrix[0] = texelFetch(uSkinningDatas, matrixId+0);
  skMatrix[1] = texelFetch(uSkinningDatas, matrixId+1);
  skMatrix[2] = texelFetch(uSkinningDatas, matrixId+2);
}

subroutine(skinning_subroutine)
void skinning_LBS(in vec4 weights, inout vec3 v, inout vec3 n) {
  /// To allow less texture fetching, and thus better memory bandwith, skinning
  /// matrices are setup as 3x4 on the host (ie. transposed and last column removed)
  /// Hence, matrix / vector multiplications are "reversed".

  // Retrieve skinning matrices
  mat3x4 jointMatrix[4];
  getSkinningMatrix(inJointIndices.x, jointMatrix[0]);
  getSkinningMatrix(inJointIndices.y, jointMatrix[1]);
  getSkinningMatrix(inJointIndices.z, jointMatrix[2]);
  getSkinningMatrix(inJointIndices.w, jointMatrix[3]);

  mat4x3 M;
  vec4 x_;

  // Position
  x_ = vec4(v, 1.0f);
  M[0] = x_ * jointMatrix[0];
  M[1] = x_ * jointMatrix[1];
  M[2] = x_ * jointMatrix[2];
  M[3] = x_ * jointMatrix[3];
  v = M * weights;

  // Normal
  x_ = vec4(n, 0.0f);
  M[0] = x_ * jointMatrix[0];
  M[1] = x_ * jointMatrix[1];
  M[2] = x_ * jointMatrix[2];
  M[3] = x_ * jointMatrix[3];
  n = M * weights;
}

///--------------------------------------------------------------------
/// BLEND SHAPE
///--------------------------------------------------------------------

void calculate_blendshape(inout vec3 v, inout vec3 n) {
  for (int i = 0; i < int(uBS_used); ++i) {
    float   weight = texelFetch(uBS_weights, i);
    int      index = texelFetch(uBS_indices, i);

    int     lut_id = gl_VertexID * int(uBS_count) + index;
    int  target_id = texelFetch(uBS_LUT, lut_id);

    // Position
    v += weight * texelFetch(uBS_data, target_id).xyz;
    // Normal [todo]
    //n += weight * texelFetch(uBS_data, 2*target_id + 1);
  }
}

///--------------------------------------------------------------------
//// MAIN
///--------------------------------------------------------------------

void main() {
  // Input
  vec3 position = inPosition;
  vec3 normal   = inNormal;

  // Processing
  calculate_skinning(position, normal);
  calculate_blendshape(position, normal);

  // Output
  gl_Position   = uModelViewProjMatrix * vec4(position, 1.0f);
  OUT.normal    = normalize(uNormalMatrix * normal);
  OUT.texCoord  = inTexCoord;
}



--

//------------------------------------------------------------------------------


-- FS

uniform sampler2D uDiffuseMap;
uniform vec3 uDiffuseColor    = vec3(1.0f);
uniform vec3 uLightDir        = normalize(vec3(-1.0f, 3.5f, 5.20f));
uniform bool uEnableTexturing = false;
uniform bool uEnableLighting  = true;

in VDataBlock {
  vec3 normal;
  vec2 texCoord;
} IN;

layout(location = 0) out vec4 fragColor;


void main() {
  vec3 color = uDiffuseColor;
  
  // Colored normal [debug]
  //color = 0.5f * (1.0f + IN.normal);

  if (uEnableTexturing) {
    color = texture(uDiffuseMap, IN.texCoord).rgb;
  }

  // Manually added sky color
  color *= vec3(0.95f, 0.92f, 0.9f);

  if (uEnableLighting) {
    color *= max(0.7f, dot(-uLightDir.xyz, IN.normal)) *
             max(0.9f, dot(+uLightDir.yzx, IN.normal)) *
             max(0.9f, dot(-uLightDir.zxy, IN.normal));
  }

  fragColor = vec4(color, 1.0f);
}
