// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------
//      A set of Fragment Shaders to post-process Hardware depth buffer.
//      The vertex shader associated with it is MapScreen.VS
//------------------------------------------------------------------------------



-- NoMSAA.FS

uniform sampler2D uDepthTex;
uniform  float uLinA;//
uniform  float uLinB;//

in VDataBlock {
  vec2 texCoord;
} IN;

out layout(location = 0) float fragDepth;

void main()
{
  float z = texture( uDepthTex, IN.texCoord).r;

  float linearDepth;
  //linearDepth = uLinB / (z - uLinA);
  //linearDepth = 1.0f / (z*uLinA + uLinB);

  float n = uLinA; //
  float f = uLinB; //
  linearDepth = (2 * n) / (f + n - z * (f - n));

  fragDepth = linearDepth;
}


--


//------------------------------------------------------------------------------

/*
-- MSAA.FS

uniform image2DMS uDepthMSImg;
uniform float uLinA;//
uniform float uLinB;//

in VDataBlock {
  vec2 texCoord;
} IN;

out layout(location = 0) float fragDepth;

void main()
{
  const int sampleId = 0;
  const ivec2 pos = ivec2( IN.texCoord * imageSize(uDepthMSImg) );
  float z = imageLoad( uDepthMSImg, pos, sampleId).r;
  fragDepth = uLinB / (z - uLinA);
}
*/

//------------------------------------------------------------------------------


-- Downsample.FS

uniform image2D uDepthImg;

in VDataBlock {
  vec2 texCoord;
} IN;

out layout(location = 0) float fragDepth;

void main()
{
  const ivec2 pos = ivec2(2 * IN.texCoord * imageSize(uDepthMSImg));
  fragDepth = imageLoad(uDepthImg, pos).r;
}
