// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef MARCHEUR_CHARACTER_H_
#define MARCHEUR_CHARACTER_H_

#include <vector>

#include "aer/aer.h"
#include "aer/animation/blend_shape.h"
#include "aer/animation/skeleton_controller.h"
#include "aer/rendering/mesh.h"
#include "aer/rendering/material.h"
#include "aer/device/program.h"
#include "aer/device/texture_buffer.h"

#include "marcheur/skm_model.h" // [wip]


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// 
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class Character {
 public:
  void init();
  void update();
  void render(const aer::Camera &camera);

 private:
  void init_shaders();
  void init_animations();

  aer::SKMModel skmModel_; //
  aer::Program  mProgram;
};

#endif  // MARCHEUR_CHARACTER_H_
