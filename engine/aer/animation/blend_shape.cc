// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/animation/blend_shape.h"

#include <vector>
#include "aer/loader/skma.h"

// =============================================================================
namespace aer {
// =============================================================================

BlendShape::BlendShape()
  : mCount(0u)
{}

//------------------------------------------------------------------------------

BlendShape::~BlendShape() {
  for (auto &tbo : mTBO) {
    tbo.buffer.release();
    tbo.texture.release();
  }
  for (auto& e : mExpressions) {
    AER_SAFE_DELETEV(e.pName);
  }
  mExpressions.clear();
}

//------------------------------------------------------------------------------

void BlendShape::init(const SKMFile &skmFile) {
  const SKMFile::TSKeyInfo *pSKeyInfos = skmFile.skey_infos();
  const SKMFile::TSKeyData *pSKeyDatas = skmFile.skey_datas();
  const U32 numBSData = skmFile.numskeydatas();

  mCount = skmFile.numskeyinfos();
  if (mCount == 0u) {
    AER_WARNING("no blend shapes loaded");
    return;
  }

  /// --- Initialize the blend shape index map
  for (U32 i = 0u; i < mCount; ++i) {
    std::string key(pSKeyInfos[i].name);
    mBSIndexMap[key] = i;
  }

  /// Setup basic expressions from blend shapes
  init_expressions();


  /// --- Setup device buffers
  U32 bytesize = 0u;

  // Updatable buffers : indices & weights [currently not used]
  /// Blend Shape's indices
  mTBO[BS_INDICES].buffer.generate();
  mTBO[BS_INDICES].buffer.bind(GL_TEXTURE_BUFFER);
  bytesize = mCount * sizeof(I32);
  mTBO[BS_INDICES].buffer.allocate(bytesize, GL_DYNAMIC_DRAW);

  /// Blend Shape's weights
  mTBO[BS_WEIGHTS].buffer.generate();
  mTBO[BS_WEIGHTS].buffer.bind(GL_TEXTURE_BUFFER);
  bytesize = mCount * sizeof(F32);
  mTBO[BS_WEIGHTS].buffer.allocate(bytesize, GL_STREAM_DRAW);

  // Immutable buffers : LUT & datas
  // Note : on GL4.4+, instead of glBufferData(..), we could use
  //        glBufferStorage(GL_TEXTURE_BUFFER, bytesize, buffer.data(), 0);
  //        to assert immutability.
  mTBO[BS_DATAS].buffer.generate();
  mTBO[BS_DATAS].buffer.bind(GL_TEXTURE_BUFFER);  
  bytesize = (1u + numBSData) * sizeof(SKMFile::TVector);
  mTBO[BS_DATAS].buffer.allocate(bytesize, GL_STATIC_DRAW);

  mTBO[BS_LUT].buffer.generate();
  mTBO[BS_LUT].buffer.bind(GL_TEXTURE_BUFFER);
  bytesize = mCount * skmFile.numvertices() * sizeof(I32);
  mTBO[BS_LUT].buffer.allocate(bytesize, GL_STATIC_DRAW);

  DeviceBuffer::Unbind(GL_TEXTURE_BUFFER);
  CHECKGLERROR();

  // Textures
  mTBO[BS_INDICES].texture.generate();
  mTBO[BS_INDICES].texture.bind();
  mTBO[BS_INDICES].texture.set_buffer(GL_R32I, mTBO[BS_INDICES].buffer);

  mTBO[BS_WEIGHTS].texture.generate();
  mTBO[BS_WEIGHTS].texture.bind();
  mTBO[BS_WEIGHTS].texture.set_buffer(GL_R32F, mTBO[BS_WEIGHTS].buffer);

  mTBO[BS_DATAS].texture.generate();
  mTBO[BS_DATAS].texture.bind();
  mTBO[BS_DATAS].texture.set_buffer(GL_RGB32F, mTBO[BS_DATAS].buffer);

  mTBO[BS_LUT].texture.generate();
  mTBO[BS_LUT].texture.bind();
  mTBO[BS_LUT].texture.set_buffer(GL_R32I, mTBO[BS_LUT].buffer);

  Texture::Unbind(GL_TEXTURE_BUFFER);
  CHECKGLERROR();


  // temp buffer used to compute the LUT buffer afterward
  std::vector<I32> pointIDToTargetID(mCount * skmFile.numpoints(), 0);

  /// Setup datas device buffer
  mTBO[BS_DATAS].buffer.bind(GL_TEXTURE_BUFFER);
  SKMFile::TVector *d_datas = nullptr;
  mTBO[BS_DATAS].buffer.map(&d_datas, GL_WRITE_ONLY);
  {
    // the first index is used by all non transformed vertices (null vector)
    d_datas[0u] = SKMFile::TVector(0.0f, 0.0f, 0.0f);

    for (U32 i = 0u, bs_id = 0u; i < numBSData; ++i) {
      const SKMFile::TSKeyInfo &skInfo = pSKeyInfos[bs_id];

      if (i >= skInfo.start + skInfo.count) {
        bs_id += 1u;
      }

      const SKMFile::TSKeyData &skData = pSKeyDatas[i];
      U32 pointBS_id = skData.pointId * mCount + bs_id;
      U32 target_id = i + 1u;
      pointIDToTargetID[pointBS_id] = target_id;

      const SKMFile::TVector &v = skData.coordRel;
      d_datas[target_id] = skData.coordRel;
    }
  }
  mTBO[BS_DATAS].buffer.unmap(&d_datas);


  /// Setup LUT device buffer
  mTBO[BS_LUT].buffer.bind(GL_TEXTURE_BUFFER);
  I32 *d_lut = nullptr;
  mTBO[BS_LUT].buffer.map(&d_lut, GL_WRITE_ONLY);
  {
    for (U32 vid = 0u; vid < skmFile.numvertices(); ++vid) {
      U32 pid = skmFile.vertices()[vid].pointId;
      for (U32 bs_id = 0u; bs_id < mCount; ++bs_id) {
        U32 src_id = pid * mCount + bs_id;
        U32 dst_id = vid * mCount + bs_id;
        d_lut[dst_id] = pointIDToTargetID[src_id];
      }
    }
  }
  mTBO[BS_LUT].buffer.unmap(&d_lut);

  DeviceBuffer::Unbind(GL_TEXTURE_BUFFER);
  CHECKGLERROR();
}

//------------------------------------------------------------------------------

void BlendShape::DEBUG_display_names() {
  BSIndexMap_t::iterator it = mBSIndexMap.begin();

  fprintf(stderr, "\n------- BlendShape index / names :\n");
  for (it = mBSIndexMap.begin(); it != mBSIndexMap.end(); ++it) {
    fprintf(stderr, "(%2d) %s\n", it->second, it->first.c_str());
  }
  fprintf(stderr, "------------------------------------\n\n");
}

//------------------------------------------------------------------------------

void BlendShape::init_expressions() {
  /// Expressions are set of individual blendshape, making a "blendshape clip".
  /// For now, there is now way to specify expressions externally to import so
  /// we just use simple expression of 1 blendshape.

  mExpressions.resize(mCount);

  U32 index = 0u;
  for (auto& pair : mBSIndexMap) {
    auto &expr = mExpressions[index++];
    expr.pName = new char[64]; //
    sprintf(expr.pName, "%s", pair.first.c_str());

    expr.clip_duration = 1.0f;
    expr.bLoop         = true;

    expr.indices.push_back(pair.second);    
  }
}

// =============================================================================
}  // namespace aer
// =============================================================================