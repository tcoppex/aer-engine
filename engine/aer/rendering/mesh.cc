// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/rendering/mesh.h"


namespace aer {

Mesh::Mesh() :
  primitive_mode_(GL_TRIANGLES),
  indices_type_(GL_UNSIGNED_INT),
  nvertices_(0u),
  nindices_(0u),
  bInitialized(false)
{}

Mesh::~Mesh() {
  vao_.release();

  for (auto &vbo : vbos_) {
    vbo.release();
  }

  if (use_indices()) {
    ibo_.release();
  }
}

void Mesh::init(U32 nBuffer, bool bUseIBO) {
  AER_ASSERT(!bInitialized);
  AER_ASSERT(nBuffer > 0);

  vao_.generate();

  vbos_.resize(nBuffer);
  for (auto &vbo : vbos_) {
    vbo.generate();
  }

  if (bUseIBO) {
    ibo_.generate();
  }

  bInitialized = true;
}

void Mesh::begin_update() {
  vao_.bind();
}

void Mesh::end_update() {
  vao_.unbind();
}

DeviceBuffer& Mesh::vbo(U32 index) {
  AER_ASSERT(index < vbo_count());
  return vbos_[index];
}

DeviceBuffer& Mesh::ibo() {
  AER_ASSERT(use_indices());
  return ibo_;
}

void Mesh::draw() const {
  vao_.bind();
  
  if (use_indices()) {
    glDrawElements(primitive_mode_, nindices_, indices_type_, 0u);
  } else {
    glDrawArrays(primitive_mode_, 0, nvertices_);
  }

  vao_.unbind();
}

void Mesh::draw_instances(const U32 count) const {
  vao_.bind();
  
  if (use_indices()) {
    glDrawElementsInstanced(primitive_mode_, nindices_, indices_type_, 0u, count);
  } else {
    glDrawArraysInstanced(primitive_mode_, 0, nvertices_, count);
  }

  vao_.unbind(); 
}

U32 Mesh::add_submesh(UPTR offset, U32 nelems, I32 tag) {
  submeshes_.push_back(SubMeshInfo_t(offset, nelems, tag));
  return submeshes_.size()-1u;
}

void Mesh::draw_submesh(U32 id) const {
  AER_ASSERT(id < submesh_count());

  vao_.bind();

  const auto &submesh = submeshes_[id];
  if (use_indices()) {
    glDrawElements(primitive_mode_,
                   submesh.nelems,
                   indices_type_,
                   reinterpret_cast<void*>(submesh.offset*sizeof(U32)));
  } else {
    glDrawArrays(primitive_mode_, submesh.offset, submesh.nelems);
  }

  vao_.unbind();
}

}  // namespace aer
