//
//


-- VS


layout(location=0) in vec4 inPositionUV;

out block {
  vec3 texCoord;
} Out;

uniform mat4 uModelViewProjMatrix;

void main() {
# define Position  (inPositionUV.xy)
# define TexCoord  (inPositionUV.zw)

  float z = -gl_InstanceID / 8.0f;
  gl_Position = uModelViewProjMatrix * vec4(Position, z, 1.0f);
  
  const int texDim = 33;
  float r = (1 + 2*gl_InstanceID) / float(2*texDim);  
  
  Out.texCoord = vec3(TexCoord, r); //
}

--

//------------------------------------------------------------------------------

-- FS

in block {
  vec3 texCoord;
} In;

out vec4 fragColor;

uniform sampler3D uDensityVolume;


void main()
{
  float d = texture( uDensityVolume, In.texCoord).r;
  d = (d<0) ? 0.25f : 0.75f;
  
  fragColor.rgb = vec3(d);
}

