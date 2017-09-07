// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "marcheur/character.h"

#include "aer/animation/blend_tree.h"
#include "aer/animation/blend_node.h"
#include "aer/loader/skma.h"
#include "aer/loader/skma_utils.h"
#include "aer/view/camera.h"


void Character::init() {
  std::string filename = DATA_DIRECTORY "models/walker/walker";
  AER_CHECK(skmModel_.load(filename + ".skm", filename + ".mat")); //

  // ----

  // Program shaders
  init_shaders();

  // Statically set the animation properties used by the character.
  // Future version should use an Action State Machine load at runtime with AI.
  init_animations();
}

void Character::update() {
  const aer::EventsHandler &ev = aer::EventsHandler::Get();

  /// Switch between Dual Quaternion & Linear blend skinning
  if (ev.key_pressed(aer::Keyboard::M)) {
    aer::SkeletonController &skl_controller = skmModel_.skeleton_controller();
    aer::SkinningMethod_t method = skl_controller.skinning_method();

    if (method == aer::SKINNING_DQB) {
      method = aer::SKINNING_LB;
      printf("Skinning : Linear Blending\n");
    } else if (method == aer::SKINNING_LB) {
      method = aer::SKINNING_DQB;
      printf("Skinning : Dual Quaternion\n");
    }
    skl_controller.set_skinning_method(method);
  }

  //---

  skmModel_.update();
}

void Character::render(const aer::Camera &camera) {  
  aer::I32 texUnit = 0;

  mProgram.activate();

  // --- Skinning on GPU using GLSL ------------------------
  aer::SkeletonController &skl_controller = skmModel_.skeleton_controller();
  skl_controller.bind_skinning_texture(texUnit);
  mProgram.set_uniform("uSkinningDatas", texUnit);
  ++texUnit;
  
  // TODO : control the subroutine handle properly
  //mProgram.set_subroutine_uniform("uSkinning", skl_controller.skinning_method());
  aer::U32 su_index = skl_controller.skinning_method();
  glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &su_index);

  CHECKGLERROR();
  // -------------------------------------------------------


  // --- Rendering -----------------------------------------
  // TODO : use custom sampler
  aer::DefaultSampler::AnisotropyRepeat().bind(texUnit);

  const aer::Matrix4x4 &viewProj = camera.view_projection_matrix();
  mProgram.set_uniform("uModelViewProjMatrix", viewProj);

  const aer::Mesh &mesh = skmModel_.mesh(); //
  for (aer::U32 i = 0u; i < mesh.submesh_count(); ++i) {
    aer::I32 material_id = mesh.submesh_tag(i);

    bool bEnableTexturing = false;
    if (material_id >= 0) {
      const auto &material = skmModel_.material(material_id); //

      if (material.has_diffuse_map()) {
        bEnableTexturing = true;
        mProgram.set_uniform("uDiffuseMap", texUnit);
        material.diffuse_map()->bind(texUnit);
      } else {
        //mProgram.set_uniform("uDiffuseColor", material.phong_attributes().diffuse);
        float c = (i+1) / static_cast<float>(mesh.submesh_count());
        mProgram.set_uniform("uDiffuseColor", aer::Vector3(c));
      }
    }
    mProgram.set_uniform("uEnableTexturing", bEnableTexturing);

    mesh.draw_submesh(i);
  }
  // -------------------------------------------------------

  mProgram.deactivate();
  CHECKGLERROR();
}

void Character::init_shaders() {
  aer::ShaderProxy &sp = aer::ShaderProxy::Get();

  // used by rendering, skinning and blend shapes
  mProgram.create();
  mProgram.add_shader(sp.get("Skinning.VS"));
  mProgram.add_shader(sp.get("Skinning.FS"));
  AER_CHECK(mProgram.link());

  CHECKGLERROR();
}

void Character::init_animations() {
  aer::BlendTree &blendtree = skmModel_.skeleton_controller().blend_tree();

  blendtree.add_leave("walking_depressed");
}
