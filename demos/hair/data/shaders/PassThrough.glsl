/*
 *          PassThrough.glsl
 *
 */

//------------------------------------------------------------------------------


-- VS


// IN
layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec3 inNormal;

// OUT
out block {
  vec3 color;
  vec3 normal;
} OUT;

// UNIFORM
uniform mat4 uMVP;
uniform vec3 uColor;

void main()
{
  gl_Position  = uMVP * inPosition;
  OUT.color    = uColor; //0.5f*(inNormal+1.0f);
  OUT.normal   = inNormal;
}


--

//------------------------------------------------------------------------------


-- FS

// IN
in block {
  vec3 color;
  vec3 normal;
} IN;

// OUT
layout(location = 0) out vec4 fragColor;

void main() {
  fragColor = vec4(IN.color, 1.0f);
}

