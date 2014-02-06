// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------
//      Generic shaders to fill deferred shading G-Buffers
//------------------------------------------------------------------------------


-- VS



uniform mat4 uModelViewProjMatrix;
uniform mat3 uNormalViewMatrix;

in layout(location = 0) vec4 inPosition;
in layout(location = 1) vec3 inNormal;
in layout(location = 2) vec2 inTexCoord;

out VDataBlock {
  vec2 texCoord;
  vec3 normalVS;
} OUT;

void main()
{
  gl_Position = uModelViewProjMatrix * inPosition;
  
  OUT.texCoord = inTexCoord;
  OUT.normalVS = normalize(inPosition.xyz);//uNormalViewMatrix * inNormal;
}


--

//------------------------------------------------------------------------------


-- FS


uniform vec3 uColor = vec3(1.0f);
uniform bool uEnableTexturing = false;
uniform sampler2D uTexture;


in VDataBlock {
  vec2 texCoord;
  vec3 normalVS;
} IN;


out layout(location = 0) vec4 outDiffuse;
out layout(location = 1) vec3 outNormal;

void main()
{
  // == Diffuse color ==
  vec4 fragDiffuse = vec4( uColor, 1.0f);   
  if (uEnableTexturing) {
    fragDiffuse = texture( uTexture, IN.texCoord);
  }

  // == Normal [View-Space] ==
  // [ TODO: Store only the XY component and retrieve the Z later ]
  // works when CullFace enables (to avoid z-fighting jaggies)
  vec3 fragNormal = (gl_FrontFacing) ? IN.normalVS : -IN.normalVS; 
  fragNormal = normalize(fragNormal);
  

  // Output  
  outDiffuse = fragDiffuse;
  outNormal  = fragNormal;

  //outDiffuse.rgb = 0.5f*fragNormal + 0.5f;
}

