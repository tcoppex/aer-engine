// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------
//        Linear Blend Skinning + BlendShape on GPU
//------------------------------------------------------------------------------


-- VS


#define NO_JOINT      0xff


// Transformation
uniform mat4          uModelViewProjMatrix;

// Skinning
uniform samplerBuffer uSkinningMatrices;

// Blend Shape
//uniform           vec4 uBS_weights;
//uniform          ivec4 uBS_indices;
uniform  samplerBuffer uBS_weights;
uniform isamplerBuffer uBS_indices;
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


void calculateShapedVertex(inout vec4 position, inout vec3 normal)
{
  for (int i = 0; i < int(uUsedBlendShape); ++i) {
    int   index  = texelFetch(uBS_indices, i); //uBS_indices[i]
    float weight = texelFetch(uBS_weights, i); //uBS_weights[i]

    int lut_idx    = gl_VertexID * int(uNumBlendShape) + index;
    int target_idx = texelFetch(uBS_LUT, lut_idx);

    position += weight * texelFetch(uBS_data, target_idx);
    //normal += weight * texelFetch(uBS_data, 2*target_idx + 1);
  }
  position.w = 1.0f;
  //normal = normalize(normal);
}

mat4 getSkinningMatrix(int jointId)
{
  if (jointId == NO_JOINT) {
    return mat4(1.0f);
  }

  const int matrixId = 4*jointId;
  mat4 skMatrix;
  skMatrix[0] = texelFetch(uSkinningMatrices, matrixId+0);
  skMatrix[1] = texelFetch(uSkinningMatrices, matrixId+1);
  skMatrix[2] = texelFetch(uSkinningMatrices, matrixId+2);
  skMatrix[3] = texelFetch(uSkinningMatrices, matrixId+3);

  return skMatrix;
}

void calculateSkinnedVertex(inout vec4 position, inout vec3 normal)
{
  ivec4 jointId = inJointIndices;
  
  vec4 weight   = vec4(inJointWeight.xyz, 0.0f);
       weight.w = 1.0f-(inJointWeight.x+inJointWeight.y+inJointWeight.z);

  if (weight.x > 0.0f)
  {
    mat4 jointMatrix[4];
    
    jointMatrix[0] = getSkinningMatrix(jointId.x);
    jointMatrix[1] = getSkinningMatrix(jointId.y);
    jointMatrix[2] = getSkinningMatrix(jointId.z);
    jointMatrix[3] = getSkinningMatrix(jointId.w);

    mat4 mPos;
    mPos[0] = jointMatrix[0] * position;
    mPos[1] = jointMatrix[1] * position;
    mPos[2] = jointMatrix[2] * position;
    mPos[3] = jointMatrix[3] * position;
    position = mPos * weight;

    vec4 baseNormal = vec4(normal, 0.0f);
    mPos[0] = jointMatrix[0] * baseNormal;
    mPos[1] = jointMatrix[1] * baseNormal;
    mPos[2] = jointMatrix[2] * baseNormal;
    mPos[3] = jointMatrix[3] * baseNormal;
    normal = normalize((mPos * weight).xyz);
  }
}

void main()
{
  vec4 position = vec4(inPosition, 1.0f);
  vec3 normal   = inNormal;

  calculateShapedVertex(position, normal);    // TODO : normals !
  calculateSkinnedVertex(position, normal);

  gl_Position   = uModelViewProjMatrix * position;
  OUT.normal    = normal;
  OUT.texCoord  = inTexCoord;
}



--

//------------------------------------------------------------------------------


-- FS


// UNIFORM
uniform sampler2D uDiffuseMap;
uniform vec3 uDiffuseColor = vec3(1.0f);
uniform vec3 uLightDir = normalize(vec3(-1.0f, 3.5f, 5.20f));
uniform bool uEnableTexturing = false;
uniform bool uEnableLighting  = true;

// IN
in VDataBlock {
  vec3 normal;
  vec2 texCoord;
} IN;

// OUT
layout(location = 0) out vec4 fragColor;


void main()
{
  vec3 color = uDiffuseColor;
  
  // Colored normal [debug]
  //color = 0.5f * (1.0f + IN.normal);
  
  if (uEnableTexturing) {
    color = texture(uDiffuseMap, IN.texCoord).rgb;
  }

  // Manually added sky color
  //color *= vec3(0.75f,0.45f,0.8f);

  if (uEnableLighting) {
    color *= max(0.7f, dot(-uLightDir, IN.normal)) *
             max(0.9f, dot(uLightDir.yzx, IN.normal)) *
             max(0.9f, dot(-uLightDir.zxy, IN.normal));
  }
  
  fragColor = vec4(color, 1.0f);
}
