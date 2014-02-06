cgc -oglsl -profile gp5vp shader.vs 


/*
#version 330

uniform mat4 uModelViewProjMatrix;

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

out VDataBlock {
  vec3 normal;
  vec2 texCoord;
} OUT;

void main()
{
  gl_Position  = uModelViewProjMatrix * inPosition;
  OUT.normal   = inNormal;
  OUT.texCoord = inTexCoord;
}
*/

# cgc version 3.0.0016, build date Feb 13 2011
# command line args: -oglsl -profile gp5vp
# source file: shader.vs
#vendor NVIDIA Corporation
#version 3.0.0.16
#profile gp5vp
#program main
#semantic uModelViewProjMatrix
#var float4 gl_Position : $vout.POSITION : HPOS : -1 : 1
#var float3 OUT.normal : $vout.ATTR0 : ATTR0 : -1 : 1
#var float2 OUT.texCoord : $vout.ATTR1 : ATTR1 : -1 : 1
#var float4x4 uModelViewProjMatrix :  : c[0], 4 : -1 : 1
#var float4 inPosition : $vin.ATTR0 : ATTR0 : -1 : 1
#var float3 inNormal : $vin.ATTR1 : ATTR1 : -1 : 1
#var float2 inTexCoord : $vin.ATTR2 : ATTR2 : -1 : 1
PARAM c[4] = { program.local[0..3] };
ATTRIB vertex_attrib[] = { vertex.attrib[0..2] };
OUTPUT result_attrib[] = { result.attrib[0..1] };
TEMP R0;
MUL.F R0, vertex.attrib[0].y, c[1];
MAD.F R0, vertex.attrib[0].x, c[0], R0;
MAD.F R0, vertex.attrib[0].z, c[2], R0;
MAD.F result.position, vertex.attrib[0].w, c[3], R0;
MOV.F result.attrib[0].xyz, vertex.attrib[1];
MOV.F result.attrib[1].xy, vertex.attrib[2];
END
# 6 instructions, 1 R-regs

