/*
 *          PassThrough.glsl
 *
 */

//------------------------------------------------------------------------------


-- VS


// IN
layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

// OUT
out block {
  vec3 color;
  vec3 normal;
  vec2 texCoord;
} OUT;

// UNIFORM
uniform mat4 uModelViewProjMatrix;
uniform mat3 uNormalViewMatrix;
uniform vec3 uColor = vec3(1.0f);

void main()
{
  gl_Position  = uModelViewProjMatrix * inPosition;
  OUT.color    = 0.5f*(normalize(inPosition.xyz)+1.0f);//uColor;
  OUT.normal   = uNormalViewMatrix * inNormal;
  OUT.texCoord = inTexCoord;
}


--

//------------------------------------------------------------------------------


-- FS

// IN
in block {
  vec3 color;
  vec3 normal;
  vec2 texCoord;
} IN;

// OUT
layout(location = 0) out vec4 fragColor;

// UNIFORM
uniform sampler2D uTexture;
uniform bool uEnableTexturing = false;

void main()
{
  fragColor = (uEnableTexturing) ? texture( uTexture, IN.texCoord) :
                                   vec4( IN.color, 1.0f);
}

