//
//    Gen Vertices
//


-- VS


layout(location = 0) in uint x6y6z6_e4e4e4; //

out block {
  vec3 v1, n1;
  vec3 v2, n2;
  vec3 v3, n3;
} Out;

const vec3 uEdgeStart[12] = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 0}, 
                             {0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {0, 0, 1}, 
                             {0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};

const vec3 uEdgeEnd[12] = {{0, 1, 0}, {1, 1, 0}, {1, 1, 0}, {1, 0, 0}, 
                           {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 0, 1}, 
                           {0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1}};

const vec3 uEdgeDir[12] = {{0, 1, 0}, {1, 0, 0}, {0, 1, 0}, {1, 0, 0}, 
                           {0, 1, 0}, {1, 0, 0}, {0, 1, 0}, {1, 0, 0}, 
                           {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};


uniform sampler3D uDensityVolume_nearest;
uniform sampler3D uDensityVolume_linear;

uniform vec3 uChunkPositionWS;
uniform float uVoxelSize;
uniform float uMargin;
uniform float uInvWindowDim;
uniform float uWindowDim;

const float kTexelSize = 1.0f / (textureSize(uDensityVolume_nearest, 0).x);


struct Vertex {
  vec3 position;
  vec3 normal;
};



Vertex place_vertex_on_edge(vec3 wsCoord, vec3 texcoord, uint edge_id) {
  Vertex v;

  const vec3 edge_start = uEdgeStart[edge_id];
  const vec3 edge_end   = uEdgeEnd[edge_id];
  const vec3 edge_dir   = uEdgeDir[edge_id];


  // Position
  float p1 = texture(uDensityVolume_nearest,
                     texcoord + edge_start * uInvWindowDim).r;
  float p2 = texture(uDensityVolume_nearest,
                     texcoord + edge_end * uInvWindowDim).r;
  float t = p1 / (p1 - p2);
  t = clamp(t, 0.0f, 1.0f);

  vec3 pos_within_cell = edge_start + t * edge_dir;  
  v.position = wsCoord + uVoxelSize * pos_within_cell;

  // Normal
  // Note : density volume needs to have a marge of at least 1 texel to be 
  //        compute correctly.
    
  const vec3 uvw = texcoord + uInvWindowDim * pos_within_cell;
  const vec2 offset = vec2(kTexelSize, 0.0f);

  vec3 gradient;
  gradient.x = texture(uDensityVolume_linear, uvw - offset.xyy).r - 
               texture(uDensityVolume_linear, uvw + offset.xyy).r;
  gradient.y = texture(uDensityVolume_linear, uvw - offset.yxy).r - 
               texture(uDensityVolume_linear, uvw + offset.yxy).r;
  gradient.z = texture(uDensityVolume_linear, uvw - offset.yyx).r - 
               texture(uDensityVolume_linear, uvw + offset.yyx).r;
  v.normal = -normalize(gradient);

  return v;
}

void main() {
  uint coords = x6y6z6_e4e4e4 & 0x7ffff;
  uint edges  = (x6y6z6_e4e4e4 >> 18) & 0xfff;

  uvec3 voxelCoord;
  voxelCoord.x = (coords >>  0) & 0x3F;
  voxelCoord.y = (coords >>  6) & 0x3F;
  voxelCoord.z = (coords >> 12) & 0x3F;

  uvec3 tri_edges;
  tri_edges.x = (edges >> 0) & 0x0F;
  tri_edges.y = (edges >> 4) & 0x0F;
  tri_edges.z = (edges >> 8) & 0x0F;


  vec3 wsCoord = uChunkPositionWS + uVoxelSize * voxelCoord;

  vec3 uvw  = uInvWindowDim * (voxelCoord + uMargin);
  // hack to prevent bit error (alternatively use a greater margin)
       uvw += 0.5f * uInvWindowDim;
       uvw *= (uWindowDim - 1) * uInvWindowDim;


  Vertex v;

  v = place_vertex_on_edge(wsCoord, uvw, tri_edges.x);
  Out.v1 = v.position;
  Out.n1 = v.normal;

  v = place_vertex_on_edge(wsCoord, uvw, tri_edges.y);
  Out.v2 = v.position;
  Out.n2 = v.normal;

  v = place_vertex_on_edge(wsCoord, uvw, tri_edges.z);
  Out.v3 = v.position;
  Out.n3 = v.normal;
}

--

//------------------------------------------------------------------------------

-- GS

layout(points) in;
layout(points, max_vertices = 3) out;

in block {
  vec3 v1, n1;
  vec3 v2, n2;
  vec3 v3, n3;
} In[];

out vec3 outPositionWS;
out vec3 outNormalWS;


void main() {
  outPositionWS = In[0].v1;
  outNormalWS   = In[0].n1;
  EmitVertex();
  
  outPositionWS = In[0].v2;
  outNormalWS   = In[0].n2;
  EmitVertex();
  
  outPositionWS = In[0].v3;
  outNormalWS   = In[0].n3;
  EmitVertex();
  
  EndPrimitive();
}

