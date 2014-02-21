// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/loader/skm_proxy.h"

#include "aer/animation/skeleton.h"
#include "aer/loader/skma.h"
#include "aer/loader/skma_utils.h"
#include "aer/rendering/mesh.h"
#include "aer/utils/path.h"


namespace aer {

SKMInfo_t* SKMProxy::load(const std::string& id) {
  SKMFile skmFile;

  std::string filename = id;// + ".skm";
  if (!skmFile.load(filename.c_str())) {
    return nullptr;
  }

  SKMInfo_t *info = new SKMInfo_t();

  /// Setup mesh
  init_mesh(skmFile, info->mesh);

  /// Setup blend shape if any
  if (skmFile.numskeyinfos() > 0u) {
    info->blendshape = new BlendShape();
    info->blendshape->init(skmFile);
  }

  /// Setup Skeleton reference name [xxx]
  if (skmFile.has_skeleton()) {
    std::string dirname = Path(filename).directory();
    info->skeleton_name = dirname + skmFile.ska_name(); //
  }

  /// Setup materials name
  info->material_ids.resize(skmFile.numfacematerials());
  for (U32 i = 0u; i < info->material_ids.size(); ++i) {
    info->material_ids[i] = skmFile.face_materials()[i].name;
  }

  return info;
}

void SKMProxy::init_mesh(const SKMFile& skmFile, Mesh& mesh) {
  const U32 nAttribs = 5u;
  mesh.init(nAttribs, true);

  init_mesh_vertices(skmFile, mesh);
  init_mesh_indices(skmFile, mesh);

  mesh.begin_update();
  {
    DeviceBuffer &vbo = mesh.vbo();
    SkinnedVertex sv;
    U32 attrib = 0u;

    VertexOffset offset(skmFile.numvertices());

    // POSITIONS
    glVertexAttribFormat(attrib, AER_ARRAYSIZE(sv.position), GL_FLOAT, GL_FALSE, 0); 
    glVertexAttribBinding(attrib, attrib);
    glBindVertexBuffer(attrib, vbo.id(), offset.position, sizeof(sv.position));
    //glVertexAttribPointer(attrib, AER_ARRAYSIZE(sv.position)​, GL_FLOAT, GL_FALSE,
                            //sizeof(sv.position), BUFFER_OFFSET(offset.position));
    glEnableVertexAttribArray(attrib);
    ++attrib;

    // NORMALS
    glVertexAttribFormat(attrib, AER_ARRAYSIZE(sv.normal), GL_FLOAT, GL_FALSE, 0); 
    glVertexAttribBinding(attrib, attrib);
    glBindVertexBuffer(attrib, vbo.id(), offset.normal, sizeof(sv.normal));
    glEnableVertexAttribArray(attrib);
    ++attrib;

    // TEXCOORDS
    glVertexAttribFormat(attrib, AER_ARRAYSIZE(sv.texCoord), GL_FLOAT, GL_FALSE, 0); 
    glVertexAttribBinding(attrib, attrib);
    glBindVertexBuffer(attrib, vbo.id(), offset.texCoord, sizeof(sv.texCoord));
    glEnableVertexAttribArray(attrib);
    ++attrib;

    if (skmFile.has_skeleton()) {
      // Joint indices (INTEGER)
      glVertexAttribIFormat(attrib, AER_ARRAYSIZE(sv.jointIndex), GL_UNSIGNED_BYTE, 0);
      glVertexAttribBinding(attrib, attrib);
      glBindVertexBuffer(attrib, vbo.id(), offset.jointIndex, sizeof(sv.jointIndex));
      glEnableVertexAttribArray(attrib);
      ++attrib;

      // Joint Weights
      glVertexAttribFormat(attrib, AER_ARRAYSIZE(sv.jointWeight), GL_FLOAT, GL_FALSE, 0); 
      glVertexAttribBinding(attrib, attrib);
      glBindVertexBuffer(attrib, vbo.id(), offset.jointWeight, sizeof(sv.jointWeight));
      glEnableVertexAttribArray(attrib);
      ++attrib;
    }

    mesh.ibo().bind(GL_ELEMENT_ARRAY_BUFFER);
  }
  mesh.end_update();

  CHECKGLERROR();
}

void SKMProxy::init_mesh_vertices(const SKMFile &skmFile, Mesh &mesh) {
  const U32 nvertices = skmFile.numvertices();
  const U32 npoints   = skmFile.numpoints();

  /// Setup DEVICE vertex buffer
  DeviceBuffer &vbo = mesh.vbo();
  vbo.bind(GL_ARRAY_BUFFER);

  // Allocate memory
  U32 buffersize = nvertices * sizeof(SkinnedVertex);
  vbo.allocate(buffersize, GL_STATIC_READ);

  // Setup vertices
  const SKMFile::TPoint*   const pPoints   = skmFile.points();
  const SKMFile::TVertex*  const pVertices = skmFile.vertices();

  void *d_vertices = nullptr;
  vbo.map(&d_vertices, GL_WRITE_ONLY);
  {
    VertexOffset offset(nvertices);
    UPTR baseoffset    = reinterpret_cast<UPTR>(d_vertices);

    /// Compute normals
    std::vector<Vector3> normals(npoints);
    skmautils::ComputeNormals(skmFile, normals);

    F32 *d_position    = reinterpret_cast<F32*>(baseoffset + offset.position);
    F32 *d_normal      = reinterpret_cast<F32*>(baseoffset + offset.normal);
    F32 *d_texCoord    = reinterpret_cast<F32*>(baseoffset + offset.texCoord);

    for (U32 vid = 0u; vid < nvertices; ++vid) {
      const U32 pointId = pVertices[vid].pointId;

      // POSITION
      *(d_position++) = pPoints[pointId].coord.X;
      *(d_position++) = pPoints[pointId].coord.Y;
      *(d_position++) = pPoints[pointId].coord.Z;

      // NORMAL
      *(d_normal++) = normals[pointId].x;
      *(d_normal++) = normals[pointId].y;
      *(d_normal++) = normals[pointId].z;

      // TEXCOORDS
      *(d_texCoord++) = pVertices[vid].U;
      *(d_texCoord++) = pVertices[vid].V;
    }

    if (skmFile.has_skeleton()) {
      /// Gather joints data
      std::vector<Vector4i> jointsindices(nvertices);
      std::vector<Vector3>  jointsweights(nvertices);
      skmautils::SetupJointsData(skmFile, jointsindices, jointsweights);

      U8  *d_jointIndex  = reinterpret_cast<U8*> (baseoffset + offset.jointIndex);
      F32 *d_jointWeight = reinterpret_cast<F32*>(baseoffset + offset.jointWeight);

      for (U32 vid = 0u; vid < nvertices; ++vid) {
        // JOINT INDICES
        *(d_jointIndex++) = jointsindices[vid].x;
        *(d_jointIndex++) = jointsindices[vid].y;
        *(d_jointIndex++) = jointsindices[vid].z;
        *(d_jointIndex++) = jointsindices[vid].w;

        // JOINT WEIGHTs
        *(d_jointWeight++) = jointsweights[vid].x;
        *(d_jointWeight++) = jointsweights[vid].y;
        *(d_jointWeight++) = jointsweights[vid].z;
      }
    }
  }
  vbo.unmap(&d_vertices);
  vbo.unbind();
}

void SKMProxy::init_mesh_indices(const SKMFile &skmFile, Mesh &mesh) {
  const U32 nfaces   = skmFile.numfaces();
  const U32 nindices = 3u * nfaces;

  /// Setup DEVICE index buffer
  DeviceBuffer &ibo = mesh.ibo();
  ibo.bind(GL_ELEMENT_ARRAY_BUFFER);

  // Allocate memory
  U32 buffersize = nindices * sizeof(U32);
  ibo.allocate(buffersize, GL_STATIC_READ);

  // Setup indices
  const SKMFile::TFace* pFaces = skmFile.faces();
  
  U32 *d_indices = nullptr;
  ibo.map(&d_indices, GL_WRITE_ONLY);
  {
    // !!
    // Supposed that faces are stored sorted by materialId
    // and that materialId are in {0, numMaterial}
    // !!
    I32 matId = static_cast<I32>(pFaces[0].materialId);
    U32 objId = 0u;
    U32 sub_nindices = 0u;

    for (U32 face_id = 0u; face_id < nfaces; ++face_id) {
      const auto &face = pFaces[face_id];

      if (matId != face.materialId) {
        UPTR offset = 3u * face_id - sub_nindices;
        mesh.add_submesh(offset, sub_nindices, matId);

        matId = face.materialId;
        sub_nindices = 0u;
        ++objId;
      }

      *(d_indices++) = face.v[0];
      *(d_indices++) = face.v[1];
      *(d_indices++) = face.v[2];
      sub_nindices += 3u;
    }
    mesh.add_submesh(nindices - sub_nindices, sub_nindices, matId);
  }
  ibo.unmap(&d_indices);
  ibo.unbind();

  /// Rendering propertie
  mesh.set_index_count(nindices);
  mesh.set_indices_type(GL_UNSIGNED_INT);
  mesh.set_primitive_mode(GL_TRIANGLES);
}

}  // namespace aer
