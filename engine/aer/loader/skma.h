// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_LOADER_SKMA_H_
#define AER_LOADER_SKMA_H_

#include <string>
#include <vector>

#include "aer/common.h"

// =============================================================================
namespace aer {
// =============================================================================

// forward declarations
class SKMFile;
class SKAFile;
class MATFile;

/**
 * @class SKMAFile
 * @brief Handles to load files associated with the SKMA file
 *
 * The SKMA file format is a chunk based format describing
 * mesh (SKM) & animation (SKA) data for a skeleton model,
 * along with materials (MAT) informations.
 *
 * This is the pure translation of the data format and it
 * should be restructured if datas are intend to be kept 
 * in memory.
 *
 * TODO: load unknown chunk types to specialized struct
 *
*/
class SKMAFile {
 public:
  SKMAFile();
  ~SKMAFile();

  bool load(const char* filename);

  const SKMFile& skmFile() const {
    AER_ASSERT(has_mesh());
    return *skmFile_;
  }
  
  const SKAFile& skaFile() const {
    AER_ASSERT(has_animation());
    return *skaFile_;
  }

  const MATFile& matFile() const {
    AER_ASSERT(has_materials());
    return *matFile_;
  }

  bool has_mesh()      const { return skmFile_ != nullptr; }
  bool has_animation() const { return skaFile_ != nullptr; }
  bool has_materials() const { return matFile_ != nullptr; }

 private:
  SKMFile *skmFile_;
  SKAFile *skaFile_;
  MATFile *matFile_;
};

// =============================================================================

#define SKMA_CHUNKHEADER_IDSIZE          20u

/**
 * @name ChunkHeader_t
 * @brief Header structure used by SKM / SKA files
 */
struct ChunkHeader_t {
  char        id[SKMA_CHUNKHEADER_IDSIZE];  //
  U32    dataSize;                     //
  U32    dataCount;                    //
  U32    flag;                         //

  ChunkHeader_t() : 
    dataSize(0u), 
    dataCount(0u), 
    flag(0u)
  {}
};

// -----------------------------------------------------------------------------
/// @name Chunk Header data id
// -----------------------------------------------------------------------------
#define SKM_HEADERID_MAIN         "SKMHEADER"
#define SKM_HEADERID_PNTS         "PNTS0000"
#define SKM_HEADERID_VERT         "VERT0000"
#define SKM_HEADERID_FACE         "FACE0000"
#define SKM_HEADERID_VMAT         "VMAT0000"  //
#define SKM_HEADERID_FMAT         "FMAT0000"
#define SKM_HEADERID_BWGT         "BWGT0000"
#define SKM_HEADERID_SKEYINFO     "SKEYINFO"
#define SKM_HEADERID_SKEYDATA     "SKEYDATA"
#define SKM_HEADERID_SKAINFO      "SKAINFO0"

#define SKA_HEADERID_MAIN         "SKAHEADER"
#define SKA_HEADERID_BONE         "BONE0000"
#define SKA_HEADERID_SEQU         "SEQU0000"
#define SKA_HEADERID_FRAM         "FRAM0000"

// =============================================================================

#define SKM_FACEMATERIAL_NAMESIZE   64u
#define SKM_SKEYINFO_NAMESIZE       32u
#define SKM_SKAINFO_NAMESIZE        32u

/**
 * @class SKMFile
 * @brief SKeletal Mesh file handler 
 */
class SKMFile {
 public:
  struct TVector {
    TVector() = default;

    TVector(F32 x, F32 y, F32 z) :
      X(x),
      Y(y),
      Z(z)
    {}

    F32  X, Y, Z;
  };

  //--

  struct TPoint {
    TVector coord; 
  };

  struct TVertex {
    U32  pointId;
    F32  U;
    F32  V;
    U16  auxTexCoordId;
    U16  materialId;
  };

  struct TFace {
    U32  v[3];
    I16    materialId;
    U16  _padding;
  };

  // [not used]
  struct TVertexMaterial {
    U32  vertexId;
    F32  value[3];
    U16  type;
    U16  auxVertexMaterialId;
  };

  struct TFaceMaterial {
    char        name[SKM_FACEMATERIAL_NAMESIZE];
    U32    type;
    U32    auxFaceMaterialId;
  };

  struct TBoneWeight {
    U32  boneId;
    U32  pointId;
    F32  weight;
  };

  struct TSKeyInfo {
    char name[SKM_SKEYINFO_NAMESIZE];
    U32  start;
    U32  count;
  };

  struct TSKeyData {
    U32  pointId;
    TVector   coordRel;
  };

  struct TSKAInfo {
    char      basename[SKM_SKAINFO_NAMESIZE];
  };


  SKMFile();
  ~SKMFile();

  bool load(const char *filename);

  const TPoint        * points()          const { return points_; }
  const TVertex       * vertices()        const { return vertices_; }
  const TFace         * faces()           const { return faces_; }
  const TFaceMaterial * face_materials()  const { return face_materials_; }
  const TBoneWeight   * bone_weights()    const { return bone_weights_; }
  const TSKeyInfo     * skey_infos()      const { return skey_infos_; }
  const TSKeyData     * skey_datas()      const { return skey_datas_; }

  U32 numpoints()        const { return numpoints_; }
  U32 numvertices()      const { return numvertices_; }
  U32 numfaces()         const { return numfaces_; }
  U32 numfacematerials() const { return numfacematerials_; }
  U32 numboneweights()   const { return numboneweights_; }
  U32 numskeyinfos()     const { return numskeyinfos_; }
  U32 numskeydatas()     const { return numskeydatas_; }

  /// @return True if a skeleton file (.ska) is linked
  bool has_skeleton() const { return numskainfos_ != 0u; }
  
  /// @return the linked skeleton filename if any, nullptr otherwise
  const char* ska_name() const { 
    if (has_skeleton()) {
      return ska_infos_[0u].basename; 
    }
    return nullptr;
  }


 private:
  TPoint           *points_;
  TVertex          *vertices_;
  TFace            *faces_;
  TVertexMaterial  *vertex_materials_;
  TFaceMaterial    *face_materials_;
  TBoneWeight      *bone_weights_;
  TSKeyInfo        *skey_infos_;
  TSKeyData        *skey_datas_;
  TSKAInfo         *ska_infos_;

  U32 numpoints_;
  U32 numvertices_;
  U32 numfaces_;
  U32 numvertexmaterials_;
  U32 numfacematerials_;
  U32 numboneweights_;
  U32 numskeyinfos_;
  U32 numskeydatas_;
  U32 numskainfos_;
};

// =============================================================================

#define SKA_BONE_NAMESIZE       32u
#define SKA_SEQUENCE_NAMESIZE   32u

/**
 * @class SKAFile
 * @brief SKeletal Animation file handler 
 */
class SKAFile {
 public:
  typedef SKMFile::TVector TVector;

  struct TQuaternion { 
    F32  W, X, Y, Z; 
  };

  struct TJoint {
    TQuaternion qRotation;
    TVector     vTranslation;
    F32    fScale;
  };

  //--

  struct TBone {
    char        name[SKA_BONE_NAMESIZE];
    U32    parentId;
    TJoint      joint;
  };

  struct TSequence {
    char      name[SKA_SEQUENCE_NAMESIZE];
    U32  startFrame;
    U32  numFrame;
    F32  animRate;
    U32  flag;
  };
  
  struct TFrame {
    TQuaternion  qRotation;
    TVector      vTranslate;
    F32     fScale;
  };


  SKAFile();
  ~SKAFile();

  bool load(const char* filename);

  const TBone     * bones()     const { return bones_;     }
  const TSequence * sequences() const { return sequences_; }
  const TFrame    * frames()    const { return frames_;    }

  U32 numbones()     const { return numbones_;     }
  U32 numsequences() const { return numsequences_; }
  U32 numframes()    const { return numframes_;    }


 private:
  TBone      *bones_;
  TSequence  *sequences_;
  TFrame     *frames_;

  U32 numbones_;
  U32 numsequences_;
  U32 numframes_;
};

// =============================================================================

#define MAT_NAME_SIZE         64u
#define MAT_TYPENAME_SIZE     32u
#define MAT_TEXTUREPATH_SIZE  64u

/**
 * @class MATFile
 * @brief MATerial file handler
 *
 * Note : 
 *  Actual implementation is very simple and does not scale well. 
 *  It will be redesign in further versions.
*/
class MATFile {
 public:
  enum MaterialType {
    TEXTURE_DIFFUSE,
    TEXTURE_SPECULAR,
    TEXTURE_BUMP,

    kNumMaterialType
  };

  struct MaterialData {
    char name[MAT_NAME_SIZE];
    char* mList[kNumMaterialType];

    MaterialData() : 
      mList{nullptr}
    {}

    ~MaterialData() {      
      for (U32 i = 0u; i < kNumMaterialType; ++i) {
        AER_SAFE_DELETEV(mList[i]);
      }
    }
  };


  MATFile() = default; //

  bool load(const char* filename);

  const char* filepath() const {
    return filepath_.c_str();
  }

  const char* material_name(U32 id) const {
    return material_datas_[id].name;
  }

  const char* material_from_name(const std::string& name,
                                 const MaterialType type) const;

  const char* material_from_id(U32 id,
                               const MaterialType type) const {
    return material_datas_[id].mList[type];
  }

  U32 count() const {
    return material_datas_.size();
  }


 private:
  std::vector<MaterialData> material_datas_; //
  std::string               filepath_;
};

// =============================================================================
}  // namespace aer
// =============================================================================

#endif  // AER_LOADER_SKMA_H_
