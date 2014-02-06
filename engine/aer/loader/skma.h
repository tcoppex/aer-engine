// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_LOADER_SKMA_H_
#define AER_LOADER_SKMA_H_

#include <string>
#include <vector>

#include "aer/common.h"


namespace aer {

class SKMFile;
class SKAFile;
class MATFile;


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// The SKMA file format is a chunk based format describing
/// mesh (SKM) & animation (SKA) data for a skeleton model,
/// along with materials (MAT) informations.
///
/// This is the pure translation of the data format and it
/// should be restructured if datas are intend to be kept 
/// in memory.
///
/// Version note :
///   Right now bones can be in both SKM & SKA files. 
///   It should be simplified in future versions.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
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
#define SKA_HEADERID_BONE         "BONE0000"  //
#define SKA_HEADERID_SEQU         "SEQU0000"
#define SKA_HEADERID_FRAM         "FRAM0000"


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///   Header structure used by SKM / SKA files
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
#define SKCHUNKHEADER_IDSIZE          20u

struct ChunkHeader_t {
  char        id[SKCHUNKHEADER_IDSIZE];     //
  aer::U32    dataSize;                     //
  aer::U32    dataCount;                    //
  aer::U32    flag;                         //

  ChunkHeader_t() : 
    dataSize(0u), 
    dataCount(0u), 
    flag(0u)
  {}
};

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
/// SKM (SKeletal Mesh) File handler
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
#define SKFACEMATERIAL_PATHNAMESIZE   64u
#define SKBONE_NAMESIZE               32u
#define SKSKINFO_NAMESIZE             32u
#define SKAINFO_BASENAMESIZE          32u

class SKMFile {
 public:
  struct TVector {
    TVector() = default;

    TVector(aer::F32 x, aer::F32 y, aer::F32 z) :
      X(x),
      Y(y),
      Z(z)
    {}

    aer::F32  X, Y, Z;
  };

  //--

  struct TPoint {
    TVector coord; 
  };

  struct TVertex {
    aer::U32  pointId;
    aer::F32  U;
    aer::F32  V;
    aer::U16  auxTexCoordId;
    aer::U16  materialId;
  };

  struct TFace {
    aer::U32  v[3];
    aer::I16    materialId;
    aer::U16  _padding;
  };

  struct TVertexMaterial {
    aer::U32  vertexId;
    aer::F32  value[3];
    aer::U16  type;
    aer::U16  auxVertexMaterialId;
  };

  struct TFaceMaterial {
    char        name[SKFACEMATERIAL_PATHNAMESIZE];
    aer::U32    type;
    aer::U32    auxFaceMaterialId;
  };

  struct TBoneWeight {
    aer::U32  boneId;
    aer::U32  pointId;
    aer::F32  weight;
  };

  struct TSKeyInfo {
    char name[SKSKINFO_NAMESIZE];
    aer::U32  start;
    aer::U32  count;
  };

  struct TSKeyData {
    aer::U32  pointId;
    TVector   coordRel;
  };

  struct TSKAInfo {
    char      basename[SKAINFO_BASENAMESIZE];
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

  const aer::U32 numpoints()        const { return numpoints_; }
  const aer::U32 numvertices()      const { return numvertices_; }
  const aer::U32 numfaces()         const { return numfaces_; }
  const aer::U32 numfacematerials() const { return numfacematerials_; }
  const aer::U32 numboneweights()   const { return numboneweights_; }
  const aer::U32 numskeyinfos()     const { return numskeyinfos_; }
  const aer::U32 numskeydatas()     const { return numskeydatas_; }

  
  const char* ska_name() const { 
    if (has_skeleton()) {
      return ska_infos_[0].basename; 
    }
    return nullptr;
  }

  const bool has_skeleton() const { return numskainfos_ > 0u; }


 private:
  TPoint*           points_;
  TVertex*          vertices_;
  TFace*            faces_;
  TVertexMaterial*  vertex_materials_;
  TFaceMaterial*    face_materials_;
  TBoneWeight*      bone_weights_;
  TSKeyInfo*        skey_infos_;
  TSKeyData*        skey_datas_;
  TSKAInfo*         ska_infos_;

  aer::U32 numpoints_;
  aer::U32 numvertices_;
  aer::U32 numfaces_;
  aer::U32 numvertexmaterials_;
  aer::U32 numfacematerials_;
  aer::U32 numboneweights_;
  aer::U32 numskeyinfos_;
  aer::U32 numskeydatas_;
  aer::U32 numskainfos_;
};


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
/// SKA (SKeletal Animation) File handler
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
#define SKA_SEQUENCE_NAMESIZE   32u

class SKAFile {
 public:
  typedef SKMFile::TVector TVector;

  struct TQuaternion { 
    aer::F32  W, X, Y, Z; 
  };

  struct TJoint {
    TQuaternion qRotation;
    TVector     vTranslation;
    aer::F32    fScale;
  };

  //--

  struct TBone {
    char        name[SKBONE_NAMESIZE];
    aer::U32    parentId;
    TJoint      joint;
  };

  struct TSequence {
    char      name[SKA_SEQUENCE_NAMESIZE];
    aer::U32  startFrame;
    aer::U32  numFrame;
    aer::F32  animRate;
    aer::U32  flag;
  };
  
  struct TFrame {
    TQuaternion  qRotation;
    TVector      vTranslate;
    aer::F32     fScale;
  };


  SKAFile();
  ~SKAFile();

  bool load(const char* filename);

  const TBone       *const bones()     const { return bones_;     }
  const TSequence   *const sequences() const { return sequences_; }
  const TFrame      *const frames()    const { return frames_;    }

  const aer::U32 numbones()     const { return numbones_;     }
  const aer::U32 numsequences() const { return numsequences_; }
  const aer::U32 numframes()    const { return numframes_;    }


 private:
  TBone*      bones_;
  TSequence*  sequences_;
  TFrame*     frames_;

  aer::U32 numbones_;
  aer::U32 numsequences_;
  aer::U32 numframes_;
};



/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
/// MAT (MATerial) File handler
///
/// Note : 
///  Actual materials handling is very simple
///  and does not scale well. It will be redesign
///  in further versions.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
#define MAT_NAME_SIZE         64u
#define MAT_TYPENAME_SIZE     32u
#define MAT_TEXTUREPATH_SIZE  64u

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
      for (aer::U32 i = 0u; i < kNumMaterialType; ++i) {
        AER_SAFE_DELETEV(mList[i]);
      }
    }
  };


  MATFile() = default; //

  bool load(const char* filename);


  const char* filepath() const {
    return filepath_.c_str();
  }

  const char* material_name(const aer::U32 id) const {
    return material_datas_[id].name;
  }

  const char* material_from_name(const std::string& name,
                                 const MaterialType type) const;

  const char* material_from_id(const aer::U32 id,
                               const MaterialType type) const {
    return material_datas_[id].mList[type];
  }

  const aer::U32 count() const {
    return material_datas_.size();
  }


 private:
  std::vector<MaterialData> material_datas_; //
  std::string               filepath_;
};

}  // namespace aer

#endif  // AER_LOADER_SKMA_H_
