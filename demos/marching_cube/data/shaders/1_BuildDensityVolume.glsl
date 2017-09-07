//
//    Build a Density 3D texture
//


-- VS


layout(location = 0) in vec4 inPositionUV;

out block {
  vec4 positionCS;
  smooth vec3 coordWS;
  int instanceID;
} Out;

uniform vec3  uChunkPositionWS;
uniform float uChunkSizeWS;           // world-space size of a chunk
uniform float uInvChunkDim;
uniform float uScaleCoord;            //uTextureRes * uInvWindowDim
uniform float uTexelSize;             //1.0f / uTextureRes;
uniform float uMargin;
uniform float uWindowDim;


void main() {
# define Position  inPositionUV.xy
# define TexCoord  inPositionUV.zw

  // in [0, 1]
  vec3 chunkcoord = vec3(TexCoord, gl_InstanceID * uTexelSize);
       chunkcoord *= uScaleCoord;

  // Change chunkcoord interval to calculate texels in the marge
  chunkcoord = (chunkcoord * uWindowDim - uMargin) * uInvChunkDim;

  vec3 ws = uChunkPositionWS + uChunkSizeWS * chunkcoord;

  Out.positionCS  = vec4(Position, 0.0f, 1.0f); // OpenGL clip-space
  Out.coordWS     = ws;
  Out.instanceID  = gl_InstanceID;
}


--

//------------------------------------------------------------------------------

-- GS

// The Geometry Shader sole purpose is to set the destination layer
// (via gl_Layer) for the 3D texture.

#define NVERTICES   3

layout(triangles) in;
layout(triangle_strip, max_vertices = NVERTICES) out;

in block {
  vec4 positionCS;
  vec3 coordWS;
  int instanceID;
} In[];

out block {
  vec3 coordWS;
} Out;


void main() {
  // Per primitive attributes
  gl_Layer = In[0].instanceID;
  
  // Per vertex attributes
# pragma unroll
  for (int i = 0; i < NVERTICES; ++i) {
    gl_Position = In[i].positionCS;
    Out.coordWS = In[i].coordWS;
    EmitVertex();
  }

  EndPrimitive();
}


--

//------------------------------------------------------------------------------

-- FS

in block {
  vec3 coordWS;
} In;

out float fragDensity;

//#include "Density.Include"
float compute_density(in vec3 ws);

void main() {
  vec3 ws = In.coordWS;
  fragDensity = compute_density(ws);
}
