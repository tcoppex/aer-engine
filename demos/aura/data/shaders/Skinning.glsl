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
uniform samplerBuffer uSkinningDatas;

// Blend Shape
uniform  samplerBuffer uBS_weights; //
uniform isamplerBuffer uBS_indices; //
uniform  samplerBuffer uBS_data;
uniform isamplerBuffer uBS_LUT;
uniform           uint uNumBlendShape;
uniform           uint uUsedBlendShape;

// Input attributes
layout(location = 0) in vec3  inPosition;
layout(location = 1) in vec3  inNormal;
layout(location = 2) in vec2  inTexCoord;
layout(location = 3) in ivec4 inJointIndices;
layout(location = 4) in vec3  inJointWeight;

// Output attributes
out VDataBlock {
  vec3 normal;
  vec2 texCoord;
} OUT;



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

void skinning_DQBS(in vec4 weights, inout vec3 v, inout vec3 n) {
  // Retrieve the dual quaternions
  mat4 Ma, Mb;
  getDualQuaternions(Ma, Mb);

  vec4 A = Ma * weights;  // real part
  vec4 B = Mb * weights;  // dual part

  float invNormA = 1.0f / length(A);
  A *= invNormA;
  B *= invNormA;

  // Position
  v += 2.0f * cross(A.xyz, cross(A.xyz, v) + A.w*v);              // Rotation
  v += 2.0f * (A.w * B.xyz - B.w * A.xyz + cross(A.xyz, B.xyz));  // Translation

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

void skinning_LBS(in vec4 weights, inout vec4 v, inout vec3 n) {
  /// To allow less texture fetching, and thus better memory bandwith, skinning
  /// matrices are setup as 3x4 on the host (ie. transposed and last column removed)
  /// Hence, Matrix / vector multiplication are "reversed".

  // Retrieve skinning matrices
  mat3x4 jointMatrix[4];
  getSkinningMatrix(inJointIndices.x, jointMatrix[0]);
  getSkinningMatrix(inJointIndices.y, jointMatrix[1]);
  getSkinningMatrix(inJointIndices.z, jointMatrix[2]);
  getSkinningMatrix(inJointIndices.w, jointMatrix[3]);

  mat4x3 M;
  
  // Position
  M[0] = v * jointMatrix[0];
  M[1] = v * jointMatrix[1];
  M[2] = v * jointMatrix[2];
  M[3] = v * jointMatrix[3];
  v.xyz = M * weights;

  // Normal
  vec4 n_ = vec4(n, 0.0f);
  M[0] = n_ * jointMatrix[0];
  M[1] = n_ * jointMatrix[1];
  M[2] = n_ * jointMatrix[2];
  M[3] = n_ * jointMatrix[3];
  n = M * weights;
}


///--------------------------------------------------------------------
/// SKINNING
///--------------------------------------------------------------------

void calculate_skinning(inout vec4 position, inout vec3 normal) {
  if (!(inJointWeight.x > 0.0f)) {
    return;
  }

  vec4 weights   = vec4(inJointWeight.xyz, 0.0f);
       weights.w = 1.0f - (weights.x + weights.y + weights.z);

  /// TODO : add manual switching from the application
#if 1
  // Dual Quaternion Blend Skinning
  skinning_DQBS(weights, position.xyz, normal);
#else
  // Linear Blend Skinning
  skinning_LBS(weights, position, normal);
#endif
}

///--------------------------------------------------------------------
/// BLEND SHAPE
///--------------------------------------------------------------------

void calculate_blendshape(inout vec3 v, inout vec3 n) {
  for (int i = 0; i < int(uUsedBlendShape); ++i) {
    float   weight = texelFetch(uBS_weights, i); //uBS_weights[i]
    int      index = texelFetch(uBS_indices, i); //uBS_indices[i]

    int     lut_id = gl_VertexID * int(uNumBlendShape) + index;
    int  target_id = texelFetch(uBS_LUT, lut_id);

    v += weight * texelFetch(uBS_data, target_id).xyz;
    //n += weight * texelFetch(uBS_data, 2*target_id + 1);
  }
}

///--------------------------------------------------------------------
//// MAIN
///--------------------------------------------------------------------

void main() {
  // Input
  vec4 position = vec4(inPosition, 1.0f);
  vec3 normal   = inNormal;

  // Processing
  calculate_skinning(position, normal);
  calculate_blendshape(position.xyz, normal);

  // Output
  gl_Position   = uModelViewProjMatrix * position;
  OUT.normal    = normalize(uNormalMatrix * normal);
  OUT.texCoord  = inTexCoord;
}



--

//------------------------------------------------------------------------------


-- FS

uniform sampler2D uDiffuseMap;
uniform vec3 uDiffuseColor = vec3(1.0f);
uniform vec3 uLightDir = normalize(vec3(-1.0f, 3.5f, 5.20f));
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
