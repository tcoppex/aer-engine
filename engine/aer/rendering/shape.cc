// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include <vector>

#include "aer/rendering/shape.h"
#include "aer/device/device_buffer.h"


namespace aer {

void Plane::init(const F32 width, const F32 height, const U32 resolution) {
  AER_ASSERT(resolution >= 2u);

  const aer::U32 nVertices = resolution*resolution;
  
  std::vector<aer::Vector3> vertices(nVertices);
  std::vector<aer::Vector3> normals(nVertices, aer::Vector3(0.0f, 1.0f, 0.0f));

  const aer::F32 ir = 1.0f / static_cast<F32>(resolution-1u);

  for (U32 y = 0u; y < resolution; ++y) {
    aer::U32 idx = y*resolution;
    aer::F32 dy = height*(0.5f - ir * y);

    for (U32 x = 0u; x < resolution; ++x, ++idx) {
      aer::F32 dx = width*(ir * x - 0.5f);
      vertices[idx] = aer::Vector3( dx, 0.0f, dy);
    }
  }

  const aer::U32 nIndices = (resolution-1u)*(2u*resolution) + resolution;
  std::vector<aer::U32> indices(nIndices);

  bool bState = false;
  aer::U32 idx = 0u;
  for (aer::U32 j = 0u; j < resolution-1; ++j)
  {
    aer::U32 dj0 = (j+0) * resolution;
    aer::U32 dj1 = (j+1) * resolution;
    
    for (aer::U32 i=0u; i<resolution; ++i)
    {
      if (bState==0) {
        indices[idx++] = dj1 + i;
        indices[idx++] = dj0 + i;
      } else {
        indices[idx++] = dj0 + resolution-(i+1u);
        indices[idx++] = dj1 + resolution-(i+1u);
      }
    }
    
    if (bState==false) {
      indices[idx++] = dj0 + resolution-1u;
      indices[idx++] = dj1 + resolution-1u;
    }
    bState = !bState;
  }


  mesh_.init(1u, true);
  
  mesh_.begin_update();
  {
    DeviceBuffer &vbo = mesh_.vbo();
    vbo.bind(GL_ARRAY_BUFFER);
    {
      U32 buffersize = vertices.size() * sizeof(vertices[0]) +
                        normals.size() * sizeof(normals[0]);
      vbo.allocate(buffersize, GL_STATIC_READ);

      IPTR offset = 0;
      glBindVertexBuffer(0, vbo.id(), 0, sizeof(vertices[0]));
      glVertexAttribFormat(POSITION, 3, GL_FLOAT, GL_FALSE, 0);
      glVertexAttribBinding(POSITION, 0);
      glEnableVertexAttribArray(POSITION);
      offset = vbo.upload(0, vertices.size() * sizeof(vertices[0]), vertices.data());

      glBindVertexBuffer(1, vbo.id(), offset, sizeof(normals[0]));
      glVertexAttribFormat(NORMAL, 3, GL_FLOAT, GL_FALSE, 0);
      glVertexAttribBinding(NORMAL, 1);
      glEnableVertexAttribArray(NORMAL);
      vbo.upload(offset, normals.size() * sizeof(normals[0]), normals.data());
    }
    vbo.unbind();

    /// Setup index datas
    DeviceBuffer &ibo = mesh_.ibo();
    ibo.bind(GL_ELEMENT_ARRAY_BUFFER);
    {
      ibo.allocate(nIndices*sizeof(indices[0]), GL_STATIC_READ);
      ibo.upload(0u, nIndices*sizeof(indices[0]), indices.data());
    }
  }
  mesh_.end_update();

  mesh_.set_index_count(nIndices);
  mesh_.set_indices_type(GL_UNSIGNED_INT);
  mesh_.set_primitive_mode(GL_TRIANGLE_STRIP);
}


//------------------------------------------------------------------------------


void Cube::init(const F32 length) {
  const F32 c = 0.5f * length;
  
  const F32 vertices[] = {
    +c, +c, +c,   +c, -c, +c,   +c, -c, -c,   +c, +c, -c, // +X
    -c, +c, +c,   -c, +c, -c,   -c, -c, -c,   -c, -c, +c, // -X
    +c, +c, +c,   +c, +c, -c,   -c, +c, -c,   -c, +c, +c, // +Y
    +c, -c, +c,   -c, -c, +c,   -c, -c, -c,   +c, -c, -c, // -Y
    +c, +c, +c,   -c, +c, +c,   -c, -c, +c,   +c, -c, +c, // +Z
    +c, +c, -c,   +c, -c, -c,   -c, -c, -c,   -c, +c, -c  // -Z
  };
  
  // TODO : store as normalized GL_BYTE or GL_INT_2_10_10_10_REV​
  const F32 normals[] = {
    +1.0f, 0.0f, 0.0f,  +1.0f, 0.0f, 0.0f,  +1.0f, 0.0f, 0.0f,  +1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f,  -1.0f, 0.0f, 0.0f,  -1.0f, 0.0f, 0.0f,  -1.0f, 0.0f, 0.0f,
    
    0.0f, +1.0f, 0.0f,  0.0f, +1.0f, 0.0f,  0.0f, +1.0f, 0.0f,  0.0f, +1.0f, 0.0f,
    0.0f, -1.0f, 0.0f,  0.0f, -1.0f, 0.0f,  0.0f, -1.0f, 0.0f,  0.0f, -1.0f, 0.0f,
    
    0.0f, 0.0f, +1.0f,  0.0f, 0.0f, +1.0f,  0.0f, 0.0f, +1.0f,  0.0f, 0.0f, +1.0f,
    0.0f, 0.0f, -1.0f,  0.0f, 0.0f, -1.0f,  0.0f, 0.0f, -1.0f,  0.0f, 0.0f, -1.0f
  };
  
  // TODO : store as GL_UNSIGNED_BYTE
  const F32 texCoords[] = {
    1.0f, 1.0f,  0.0f, 1.0f,  0.0f, 0.0f,  1.0f, 0.0f,
    1.0f, 1.0f,  0.0f, 1.0f,  0.0f, 0.0f,  1.0f, 0.0f,
    1.0f, 1.0f,  0.0f, 1.0f,  0.0f, 0.0f,  1.0f, 0.0f,
    1.0f, 1.0f,  0.0f, 1.0f,  0.0f, 0.0f,  1.0f, 0.0f,
    1.0f, 1.0f,  0.0f, 1.0f,  0.0f, 0.0f,  1.0f, 0.0f,
    1.0f, 1.0f,  0.0f, 1.0f,  0.0f, 0.0f,  1.0f, 0.0f    
  };
  
  const U8 indices[] = {
    0, 1, 2, 0, 2, 3,
    4, 5, 6, 4, 6, 7,
    8, 9, 10, 8, 10, 11,
    12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19,
    20, 21, 22, 20, 22, 23
  };
  
  //const U32 nVertices = AER_ARRAYSIZE(vertices) / 3;
  const U32 nIndices  = AER_ARRAYSIZE(indices);
  
  mesh_.init(1u, true);
  
  mesh_.begin_update();
    /// Setup vertex datas
    DeviceBuffer &vbo = mesh_.vbo();
    U32 buffersize = sizeof(vertices) + sizeof(normals) + sizeof(texCoords);
    
    vbo.bind(GL_ARRAY_BUFFER);
    vbo.allocate(buffersize, GL_STATIC_READ);
    {
#if 1
      IPTR offset = 0u;
      glBindVertexBuffer(POSITION, vbo.id(), offset, 3*sizeof(vertices[0]));
      glVertexAttribFormat(POSITION, 3, GL_FLOAT, GL_FALSE, 0);
      offset = vbo.upload(offset, sizeof(vertices), vertices);

      glBindVertexBuffer(NORMAL, vbo.id(), offset, 3*sizeof(normals[0]));
      glVertexAttribFormat(NORMAL, 3, GL_FLOAT, GL_FALSE, 0);
      offset = vbo.upload(offset, sizeof(normals), normals);

      glBindVertexBuffer(TEXCOORD, vbo.id(), offset, 2*sizeof(texCoords[0]));
      glVertexAttribFormat(TEXCOORD, 2, GL_FLOAT, GL_FALSE, 0);
      offset = vbo.upload(offset, sizeof(texCoords), texCoords);

      // TODO : unroll
      for (U32 i=0u; i<3u; ++i) {
        glVertexAttribBinding(i, i);
        glEnableVertexAttribArray(i);
      }
#else
    aer::UPTR offset = 0;

#define BUFFER_OFFSET  (const GLvoid*)(offset)
    // Positions
    glEnableVertexAttribArray(0);
    glVertexAttribPoI32er(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET);
    offset = vbo.upload(offset, sizeof(vertices), vertices);

    // Normals
    glEnableVertexAttribArray(1);
    glVertexAttribPoI32er(1, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET);
    offset = vbo.upload(offset, sizeof(normals), normals);

    // TexCoords
    glEnableVertexAttribArray(2);
    glVertexAttribPoI32er(2, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET);
    offset = vbo.upload(offset, sizeof(texCoords), texCoords);

#undef BUFFER_OFFSET
#endif

    AER_CHECK(offset == buffersize);
    }
    vbo.unbind();

    /// Setup index datas
    DeviceBuffer &ibo = mesh_.ibo();
    ibo.bind(GL_ELEMENT_ARRAY_BUFFER);
    ibo.allocate(sizeof(indices), GL_STATIC_READ);
    ibo.upload(0u, sizeof(indices), indices);
  mesh_.end_update();

  // Properties needed for rendering
  mesh_.set_index_count(nIndices);
  mesh_.set_primitive_mode(GL_TRIANGLES);
  mesh_.set_indices_type(GL_UNSIGNED_BYTE);

/*
  setAttrib( POSITION, vertices, 3, sizeof(vertices)/3); //format, bUpdatable
  setAttrib( TEXCOORD, texCoords, 2, sizeof(texCoords)/2);
  setAttrib( NORMAL, normals, 3, sizeof(normals)/3);
  setIndices( GL_TRIANGLES, indices, nIndices);  
  upload( aer::DeviceBuffer::STATIC_DRAW, SINGLE_BUFFER);
*/
}

void SphereRaw::init(const F32 radius, const U32 resolution) {
  const U32 nVertices = 2 * resolution * (resolution + 2u);
  std::vector<F32> vertices(3*nVertices);

  F32 theta2, phi;    // next theta angle, phi angle
  F32 ct, st;         // cos(theta), sin(theta)
  F32 ct2, st2;       // cos(next theta), sin(next theta)
  F32 cp, sp;         // cos(phi), sin(phi)

  const F32 Pi    = M_PI;
  const F32 TwoPi = 2.0f * Pi;
  const F32 Delta = 1.0f / resolution;

  ct2 = 0.0f; st2 = -1.0f;

  /* Create a sphere from bottom to top (like a spiral) as a tristrip */
  U32 id = 0u;
  for (U32 j = 0; j < resolution; ++j) {    
    ct = ct2;
    st = st2;

    theta2 = ((j+1) * Delta - 0.5f) * Pi;
    ct2 = glm::cos(theta2);
    st2 = glm::sin(theta2);

    vertices[id++] = radius * (ct);
    vertices[id++] = radius * (st);
    vertices[id++] = 0.0f;

    for (U32 i = 0u; i < resolution + 1u; ++i) {
      phi = TwoPi * i * Delta;
      cp = glm::cos(phi);
      sp = glm::sin(phi);

      vertices[id++] = radius * (ct2 * cp);
      vertices[id++] = radius * (st2);
      vertices[id++] = radius * (ct2 * sp);

      vertices[id++] = radius * (ct * cp);
      vertices[id++] = radius * (st);
      vertices[id++] = radius * (ct * sp);
    }
    vertices[id++] = radius * (ct2);
    vertices[id++] = radius * (st2);
    vertices[id++] = 0.0f;
  }


  mesh_.init(1u, false);

  mesh_.begin_update();
    /// Setup vertex datas
    DeviceBuffer &vbo = mesh_.vbo();
    U32 buffersize = vertices.size() * sizeof(vertices[0]);

    vbo.bind(GL_ARRAY_BUFFER);
    vbo.allocate(buffersize, GL_STATIC_READ);
    {
      glBindVertexBuffer(0, vbo.id(), 0, 3*sizeof(vertices[0]));
      glVertexAttribFormat(POSITION, 3, GL_FLOAT, GL_FALSE, 0);
      glVertexAttribBinding(POSITION, 0);
      glEnableVertexAttribArray(POSITION);

      vbo.upload(0, buffersize, vertices.data());
    }
    vbo.unbind();
  mesh_.end_update();

  mesh_.set_vertex_count(nVertices);
  mesh_.set_primitive_mode(GL_TRIANGLE_STRIP);
}

void Dome::init(const F32 radius, const U32 resolution) {
  const U32 nvertices = (resolution + 2u) * resolution;
  std::vector<F32> vertices(3u * nvertices);
  std::vector<F32> texCoords(2u * nvertices);
  
  
  F32 theta2, phi;    // next theta angle, phi angle
  F32 ct, st;         // cos(theta), sin(theta)
  F32 ct2, st2;       // cos(next theta), sin(next theta)
  F32 cp, sp;         // cos(phi), sin(phi)
  
  const F32 Pi    = M_PI;
  const F32 TwoPi = 2.0f * Pi;
  const F32 Delta = 1.0f / static_cast<F32>(resolution);

  ct2 = 1.0f;
  st2 = 0.0f;

  // Create a sphere from bottom to top (like a spiral) as a tristrip
  I32 id = 0;
  for (I32 j = resolution/2; j < static_cast<I32>(resolution); ++j) {
    ct = ct2;
    st = st2;
    
    theta2 = ((j+1) * Delta - 0.5f) * Pi;
    ct2 = glm::cos(theta2);
    st2 = glm::sin(theta2);
    
    vertices[3*id + 0] = radius * (ct);
    vertices[3*id + 1] = radius * (st);
    vertices[3*id + 2] = 0.0f;
    texCoords[2*id + 0] = 0.5f * ct;
    texCoords[2*id + 1] = 0.0f;
    ++id;

    for (I32 i = 0; i < static_cast<I32>(resolution)+1; ++i) {
      phi = TwoPi * i * Delta;
      cp = glm::cos(phi);
      sp = glm::sin(phi);

      vertices[3*id + 0] = radius * (ct2 * cp);
      vertices[3*id + 1] = radius * (st2);
      vertices[3*id + 2] = radius * (ct2 * sp);
      texCoords[2*id + 0] = 0.5f * ct2 * cp;
      texCoords[2*id + 1] = 0.5f * ct2 * sp;
      ++id;

      vertices[3*id + 0] = radius * (ct * cp);
      vertices[3*id + 1] = radius * (st);
      vertices[3*id + 2] = radius * (ct * sp);
      texCoords[2*id + 0] = 0.5f * ct * cp;
      texCoords[2*id + 1] = 0.5f * ct * sp;
      ++id;
    }
    vertices[3*id + 0] = radius * (ct2);
    vertices[3*id + 1] = radius * (st2);
    vertices[3*id + 2] = 0.0f;
    texCoords[2*id + 0] = 0.5f * ct2;
    texCoords[2*id + 1] = 0.0f;
    ++id;
  }


  mesh_.init(1u, false);
  mesh_.begin_update();
    /// Setup vertex datas
    DeviceBuffer &vbo = mesh_.vbo();
    U32 vertices_size = 3 * sizeof(vertices[0]);
    U32 texcoord_size = 2 * sizeof(texCoords[0]);
    U32 buffersize = nvertices * (vertices_size + texcoord_size);

    vbo.bind(GL_ARRAY_BUFFER);
    vbo.allocate(buffersize, GL_STATIC_READ);
    {
      IPTR offset = 0u;
      aer::U32 attrib = 0;

      glBindVertexBuffer(attrib, vbo.id(), offset, vertices_size);
      glVertexAttribFormat(attrib, 3, GL_FLOAT, GL_FALSE, 0);
      offset = vbo.upload(offset, nvertices * vertices_size, vertices.data());
      ++attrib;

      glBindVertexBuffer(attrib, vbo.id(), offset, texcoord_size);
      glVertexAttribFormat(attrib, 2, GL_FLOAT, GL_FALSE, 0);
      offset = vbo.upload(offset, nvertices * texcoord_size, texCoords.data());
      ++attrib;

      for (aer::U32 i = 0u; i < attrib; ++i) {
        glVertexAttribBinding(i, i);
        glEnableVertexAttribArray(i);        
      }
    }
    vbo.unbind();
  mesh_.end_update();

  mesh_.set_vertex_count(nvertices);
  mesh_.set_primitive_mode(GL_TRIANGLE_STRIP);
}


}  // namespace aer

