// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AURA_SKYDOME_H_
#define AURA_SKYDOME_H_

#include "aer/aer.h"
#include "aer/device/program.h"
#include "aer/device/texture.h"
#include "aer/rendering/shape.h"


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// SkyDome with animated sky procedurally generated.
/// Parameters : color, clouds' speed.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class SkyDome {
 public:
  const aer::U32 kDefaultTextureResolution = 512u;
  const aer::U32 kDefaultMeshResolution    = 16u;
  //const aer::U32 kDefaultMeshRadius        = 150u;

  SkyDome() = default;
  ~SkyDome();

  bool init();
  void render(const aer::Camera &camera);


 private:
  bool init_texture();

  aer::Program    mProgram;
  aer::Texture2D  mTexture;
  aer::Dome       mMesh;

  struct {
    aer::Matrix4x4  model_matrix;
    aer::Vector3    color;
    aer::F32        speed;
  } mAttribs;


  DISALLOW_COPY_AND_ASSIGN(SkyDome);
};

#endif  // AURA_SKYDOME_H_
