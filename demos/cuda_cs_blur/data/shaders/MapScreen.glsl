/*
 *          MapScreen.glsl
 *
 *    references : 
 *      http://altdevblogaday.com/2011/08/08/interesting-vertex-shader-trick/
 */

//------------------------------------------------------------------------------


-- VS


out VDataBlock {
  vec2 texCoord;
} OUT;

void main() {
  OUT.texCoord.s = (gl_VertexID << 1) & 2;
  OUT.texCoord.t = gl_VertexID & 2;

  gl_Position = vec4(2.0f * OUT.texCoord - 1.0f, 0.0f, 1.0f);
}


--

//------------------------------------------------------------------------------


-- FS

uniform sampler2D uSceneTex;

in VDataBlock {
  vec2 texCoord;
} IN;


layout(location = 0) out vec4 fragColor;

void main() {
  vec4 diffuse = texture(uSceneTex, IN.texCoord);
  fragColor = diffuse;
}
