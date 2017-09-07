/*
 *          PassThrough.glsl
 *
 */

//------------------------------------------------------------------------------


-- VS


// IN
layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec3 inNormal;

out vec3 vNormal;

uniform mat4 uModelViewProjMatrix;


void main()
{
  gl_Position = uModelViewProjMatrix * inPosition;
  vNormal = inNormal;
}


--

//------------------------------------------------------------------------------


-- FS

in vec3 vNormal;

layout(location = 0) out vec4 fragColor;

uniform vec3 uColor = vec3( 1.0f, 0.0f, 0.0f);


void main()
{
  fragColor.rgb = uColor;
}

