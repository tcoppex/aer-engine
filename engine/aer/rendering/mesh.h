// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_RENDERING_MESH_H_
#define AER_RENDERING_MESH_H_

#include <vector>

#include "aer/common.h"
#include "aer/rendering/drawable.h"
#include "aer/device/vertex_array.h"
#include "aer/device/device_buffer.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Represent a mesh on the device
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class Mesh : public Drawable {
 public:
  Mesh();
  ~Mesh();

  /// Generate the number of vertex buffer needed by the mesh
  /// + a index buffer if needed.
  void init(U32 nBuffer, bool bUseIBO);


  // Set the number of vertex sent to the device
  void set_vertex_count(U32 nvertices) {
    nvertices_ = nvertices;
  }

  /// Set the number of indices sent to the device
  void set_index_count(U32 nindices) {
    AER_CHECK(use_indices());
    nindices_ = nindices;
  }

  // Specify how the vertices are rendered, default to GL_TRIANGLES
  void set_primitive_mode(GLenum mode) {
    primitive_mode_ = mode;
  }

  // Specify the internal type of indices, default to GL_UNSIGNED_INT
  void set_indices_type(GLenum type) {
    indices_type_ = type;
  }

  // Geometry attributes updates must be processed between those call
  // to be valid.
  void begin_update();
  void end_update();

  // Return a vbo given by index, fails if index is out of bound
  DeviceBuffer& vbo(U32 index = 0u);

  // Return the ibo, fails if it does not exist.
  DeviceBuffer& ibo();

  // Draw the whole mesh
  void draw() const override;

  // Draw the whole mesh n-time
  void draw_instances(const U32 count) const override;

  // Add a submesh
  // - offset & nelems are relatives to indices OR vertices
  // depending wether an index buffer is provided or not.
  // - tag is optional and can be used to mark submesh properties (eg. materials)
  // - Return the submesh's index
  U32 add_submesh(UPTR offset, U32 nelems, I32 tag=0);

  // Draw a single sub-mesh
  void draw_submesh(U32 id) const;

  // Return the number of vbo
  U32 vbo_count() const { return vbos_.size(); }

  // Return a submesh's tag
  I32 submesh_tag(U32 id) const { return submeshes_[id].tag; }

  // Return the number of submesh
  U32 submesh_count() const { return submeshes_.size(); }

  // Return true if index buffer is use for rendering
  bool use_indices() const { return ibo_.is_generated(); }

 private:
  struct SubMeshInfo_t {
    SubMeshInfo_t(UPTR offset, U32 nelems, I32 tag) :
      offset(offset),
      nelems(nelems),
      tag(tag)
    {}

    UPTR offset;
    U32  nelems;
    I32  tag;
  };

  VertexArray   vao_;                           // Vertex array
  DeviceBuffer  ibo_;                           // Indices buffer
  std::vector<DeviceBuffer> vbos_;              // Collection of vertex buffer
    
  GLenum  primitive_mode_;                      // How to render the vertices
  GLenum  indices_type_;                        // Indices internal type
  U32     nvertices_;                           // Total number of vertices
  U32     nindices_;                            // Total number of indices
  std::vector<SubMeshInfo_t> submeshes_;        // Collection of submesh infos

  bool bInitialized;
};
  
}  // namespace aer

#endif  // AER_RENDERING_MESH_H_

#if 0
mesh.begin_update();
  mesh.vbo().allocate(bytesize, GL_STATIC_READ);
  mesh.vbo().upload(offset, bytesize, data);
  //mesh.vbo().map();
  //mesh.vbo().unmap();
  mesh.set_vertex_count(nVertex);

  glBindVertexBuffer(0, mesh.vbo().id(), baseOffset, sizeof(Vertex));
  glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, offsetof(position, Vertex));
  glVertexAttribFormat(1, 3, GL_FLOAT, GL_FALSE, offsetof(normal, Vertex));
  glVertexAttribFormat(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, offsetof(color, Vertex));
  glVertexAttribIFormat(3, 4, GL_UNSIGNED_BYTE, offsetof(index, Vertex));

  for (U32 i = 0u; i < nAttribs; ++i) {
    glVertexAttribBinding(i, 0); //attrib_binding[i];
    glEnableVertexAttribArray(i)
  }
mesh.end_update();
#endif