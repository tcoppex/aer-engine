// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aura/skydome.h"

#include "aer/device/framebuffer.h"
#include "aer/view/camera.h"
#include "aer/rendering/mapscreen.h"

// =============================================================================

SkyDome::~SkyDome() {
  mProgram.release();
  mTexture.release();
}

// -----------------------------------------------------------------------------

bool SkyDome::init() {
  // Program
  aer::ShaderProxy &sp = aer::ShaderProxy::Get();

  mProgram.create();
  mProgram.add_shader(sp.get("SkyDome.VS"));
  mProgram.add_shader(sp.get("SkyDome.FS"));
  if (!mProgram.link()) {
    return false;
  }

  // Sky textures
  if (!init_texture()) {
    return false;
  }

  // Dome mesh
  aer::F32 dome_radius = 150.0f;
  mMesh.init(1.0, kDefaultMeshResolution);

  // Rendering parameters
  mAttribs.model_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(dome_radius));
  mAttribs.color        = aer::Vector3(0.75f,0.45f,0.8f);
  mAttribs.speed        = 0.0035f;

  return true;
}

// -----------------------------------------------------------------------------

void SkyDome::render(const aer::Camera &camera) {
  aer::opengl::StatesInfo glstates = aer::opengl::PopStates();
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glDepthMask(GL_FALSE);

  // Center the dome to the camera
  aer::Matrix4x4 model = glm::translate(glm::mat4(1.0f), camera.position());
                 model = model * mAttribs.model_matrix;
  aer::Matrix4x4 mvp = camera.view_projection_matrix() * model;

  // Clouds motion speed
  aer::F32 sky_clock = aer::GlobalClock::Get().application_time(aer::SECOND);
           sky_clock *= mAttribs.speed;


  mProgram.activate();

  mProgram.set_uniform("uModelViewProjMatrix", mvp);
  mProgram.set_uniform("uSkyColor", mAttribs.color);
  mProgram.set_uniform("uSkyClock", sky_clock);
  mProgram.set_uniform("uSkyTex", 0);
  aer::DefaultSampler::LinearRepeat().bind(0);
  mTexture.bind(0);

  mMesh.draw();

  mTexture.unbind();
  aer::Sampler::Unbind(0);
  aer::Program::Deactivate();

  aer::opengl::PushStates(glstates);
  CHECKGLERROR();
}

// -----------------------------------------------------------------------------

bool SkyDome::init_texture() {
  mTexture.generate();
  mTexture.bind();
  mTexture.allocate(GL_RGB, kDefaultTextureResolution, kDefaultTextureResolution);
  mTexture.unbind();

  /// RenderTexture program
  aer::ShaderProxy &sp = aer::ShaderProxy::Get();
  aer::Program pgm;

  pgm.create();
  pgm.add_shader(sp.get("MapScreen.VS"));
  pgm.add_shader(sp.get("SkyDome.RenderTexture.FS"));
  pgm.add_shader(sp.get("Noise.Include", GL_FRAGMENT_SHADER));

  if (!pgm.link()) {
    pgm.release();
    return false;
  }

  /// RenderTexture buffer
  aer::Framebuffer rt;
  rt.generate();
  rt.bind(GL_FRAMEBUFFER);
  rt.attach_color(&mTexture, GL_COLOR_ATTACHMENT0);

  if (!aer::Framebuffer::CheckStatus()) {
    pgm.release();
    rt.release();
    return false;
  }

  /// Generate !
  aer::opengl::StatesInfo glstates = aer::opengl::PopStates();

  glDrawBuffer(GL_COLOR_ATTACHMENT0);

  glDisable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);
  glViewport(0, 0, kDefaultTextureResolution, kDefaultTextureResolution);
  glClear(GL_COLOR_BUFFER_BIT);

  pgm.activate();
  pgm.set_uniform("uEnableTiling", true);
  pgm.set_uniform("uTileRes", aer::Vector3(kDefaultTextureResolution));
  aer::I32 seed = 456789 * (rand() / static_cast<aer::F32>(RAND_MAX));
  pgm.set_uniform("uPermutationSeed", seed);

  aer::MapScreen::Draw();

  aer::Program::Deactivate();
  aer::Framebuffer::Unbind();
  pgm.release();
  rt.release();

  aer::opengl::PushStates(glstates);

  CHECKGLERROR();

  return true;
}

// =============================================================================
