// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aer/loader/skma_utils.h"
#include "aer/loader/skma.h"


namespace aer {
namespace skmautils {

void ComputeNormals(const SKMFile &skmFile, 
                    std::vector<aer::Vector3> &normals)
{
  const SKMFile::TPoint*   const pPoints   = skmFile.points();
  const SKMFile::TVertex*  const pVertices = skmFile.vertices();
  const SKMFile::TFace*    const pFaces    = skmFile.faces();


  for (aer::U32 face_id = 0u; face_id < skmFile.numfaces(); ++face_id) {
    const auto &face = pFaces[face_id];

    const aer::U32 i1 = pVertices[face.v[0]].pointId;
    const aer::U32 i2 = pVertices[face.v[1]].pointId;
    const aer::U32 i3 = pVertices[face.v[2]].pointId;

    const SKMFile::TPoint &p1 = pPoints[i1];
    const SKMFile::TPoint &p2 = pPoints[i2];
    const SKMFile::TPoint &p3 = pPoints[i3];

    aer::Vector3 A(p1.coord.X - p2.coord.X,
                   p1.coord.Y - p2.coord.Y,
                   p1.coord.Z - p2.coord.Z);
    A = glm::normalize(A);

    aer::Vector3 B(p3.coord.X - p2.coord.X,
                   p3.coord.Y - p2.coord.Y,
                   p3.coord.Z - p2.coord.Z);
    B = glm::normalize(B);

    aer::Vector3 nor = glm::cross(A, B);

    normals[i1] += nor;
    normals[i2] += nor;
    normals[i3] += nor;
  }

  for (aer::U32 j = 0u; j < normals.size(); ++j) {
    normals[j] = glm::normalize(normals[j]);
  }
}


void SetupJointsData(const SKMFile &skmFile, 
                     std::vector<aer::Vector4i> &indices,
                     std::vector<aer::Vector3>  &weights) 
{
  const SKMFile::TVertex*     const pVertices    = skmFile.vertices();
  const SKMFile::TBoneWeight* const pBoneWeights = skmFile.bone_weights();


  // Structure used to link points with their boneweights 
  // in order to facilitate skinned vertex buffer creation (like a LUT).
  struct LUTBoneWeight_t {
    aer::U32 boneweight_id;
    aer::U32 count;
    LUTBoneWeight_t() : boneweight_id(0u), count(0u) {}
  };

  LUTBoneWeight_t *point_to_boneweights = new LUTBoneWeight_t[skmFile.numpoints()];

  aer::I32 last_pointid = -1;
  for (aer::U32 bwid = 0u; bwid < skmFile.numboneweights(); ++bwid) {
    aer::I32 pointid = pBoneWeights[bwid].pointId;
    LUTBoneWeight_t &lut = point_to_boneweights[pointid];
    
    if (last_pointid == pointid) {
      lut.count += 1u;
    } else {
      last_pointid = pointid;
      lut.boneweight_id = bwid;
      lut.count = 1u;
    }
  }

  aer::U32 count_null_weights = 0u;
  for (aer::U32 vid = 0u; vid < skmFile.numvertices(); ++vid) {
    const aer::U32 pointId = pVertices[vid].pointId;
    const aer::U32 bwid    = point_to_boneweights[pointId].boneweight_id;
    const aer::U32 nWeight = point_to_boneweights[pointId].count;

    aer::F32 sum_weights = 0.0f;
    for (aer::U32 k = 0u; k < nWeight; ++k) {
      if (k >= 4) {
        AER_CHECK("Warning: A vertex is linked to more than 4 bones." && 0);
        break;
      }

      indices[vid][k] = pBoneWeights[bwid + k].boneId;
      if (k < 3) {
        weights[vid][k] = pBoneWeights[bwid + k].weight;
      }
      sum_weights += pBoneWeights[bwid + k].weight;
    }

#if AER_DEBUG
    // -- Weights incoherence checking --
    const aer::F32 eps = 1.0e-3;
    if (sum_weights == 0.0f) {
      ++count_null_weights;
    } else if (abs(sum_weights - 1.0f) > eps) {
      printf("v%6u should be normalized (total weight : %f).\n", vid, sum_weights);
    }
#endif
  }
  AER_DEBUG_CODE(printf("null weights : %u\n", count_null_weights));

  AER_SAFE_DELETEV(point_to_boneweights);
}

}  // namespace skmautils
}  // namespace aer
