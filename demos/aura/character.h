// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AURA_CHARACTER_H_
#define AURA_CHARACTER_H_

#include <vector>

#include "aer/aer.h"
#include "aer/animation/blend_shape.h"
#include "aer/animation/skeleton_controller.h"
#include "aer/rendering/mesh.h"
#include "aer/rendering/material.h"
#include "aer/device/program.h"
#include "aer/device/texture_buffer.h"

#include "aura/skm_model.h" // [wip]

// =============================================================================

/**
 * @class Character
 * @brief Handle an animated character 
 *
 * @warning first draft
*/
class Character {
public:
  void init();
  void update();
  void render(const aer::Camera &camera);

private:
  void init_shaders();
  void init_animations();
  void init_blendshapes();

  aer::SKMModel skmModel_; //
  aer::Program  mProgram;
};

// =============================================================================

#endif  // AURA_CHARACTER_H_
