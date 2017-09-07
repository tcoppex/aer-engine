// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------
//      Display the origin axis of a space (for debugging)
//------------------------------------------------------------------------------



-- VS


layout(location=0) in vec4 inPosition;

out VDataBlock {
  vec4 position;
} OUT;

void main() {
  OUT.position = inPosition;
}


--

//------------------------------------------------------------------------------

-- GS


uniform mat4 uMVP;


in VDataBlock {
  vec4 position;
} IN[];

out GDataBlock {
  vec3 color;
} OUT;

layout(points) in;
layout(line_strip, max_vertices=6) out;

void emitAxis(vec3 axis) {
  const float s = 0.5f;
  
  OUT.color = axis;
  gl_Position = uMVP * IN[0].position; 
  EmitVertex();
  
  OUT.color = axis;
  gl_Position = uMVP * vec4( IN[0].position.xyz + s*axis, 1.0f);
  EmitVertex();
}

void main() {
  emitAxis( vec3( 1.0f, 0.0f, 0.0f) );
  EndPrimitive();
  emitAxis( vec3( 0.0f, 1.0f, 0.0f) );
  EndPrimitive();
  emitAxis( vec3( 0.0f, 0.0f, 1.0f) );  
  EndPrimitive();
}


--

//------------------------------------------------------------------------------

-- FS


in GDataBlock {
  vec3 color;
} IN;

out vec3 fragColor;


void main() {
  fragColor = IN.color;
}

