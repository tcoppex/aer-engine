// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
// -----------------------------------------------------------------------------


#include "variance_shadow_mapping/application.h"


Application::Application(int argc, char* argv[]) :
  aer::Application(argc, argv),
  mCamera(nullptr)
{}

Application::~Application() {
  AER_SAFE_DELETE(mCamera);
}

void Application::init() {
  /// Window
  aer::Display_t display(1280, 720);
  display.msaa_level = 4u;
  create_window(display, "Variance Shadow Mapping");

  // Enable fps control
  set_fps_control(true);
  set_fps_limit(60u);

  /// Camera
  aer::View view(aer::Vector3(0.0f, 10.0f, 7.0f),
                 aer::Vector3(0.0f, 9.0f, 0.0f),
                 aer::Vector3(0.0f, 1.0f, 0.0f));

  aer::F32 ratio = window().display().aspect_ratio();
  aer::Frustum frustum(60.0f, ratio, 0.1f, 1000.0f);

  mCamera = new aer::FreeCamera(view, frustum);
  mCamera->set_motion_factor(0.20f);
  mCamera->set_rotation_factor(0.15f);
  mCamera->enable_motion_noise(false);

  /// OpenGL settings
  glClearColor(0.25f, 0.25f, 0.25f, 1.0f);

  /// The post-process textures & FBO must be set as MULTISAMPLE
  /// for this to works
  if (display.msaa_level > 1u) {
    glEnable(GL_MULTISAMPLE);
  } else {
    glDisable(GL_MULTISAMPLE);
  }

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);

  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  // Application settings
  init_scene();
  init_shadow();
}

namespace {
aer::F32 Animate(const aer::F32 start_pos,
                 const aer::F32 end_pos,
                 const aer::F32 duration_sec) {
  aer::F32 t = aer::GlobalClock::Get().application_time(aer::SECOND);
  t = fmodf(t, duration_sec) / duration_sec;
  t = 0.5f + 0.5f*sin(t * 2.0f * M_PI);
  return start_pos + t*(end_pos - start_pos);
}
}

void Application::frame() {
  const aer::EventsHandler &ev  = aer::EventsHandler::Get();
        aer::GlobalClock &clock = aer::GlobalClock::Get();


  /// Exit the application
  if (ev.key_pressed(aer::Keyboard::Escape)) {
    quit();
  }

  mCamera->update();
  
  aer::Vector3 position(00.0f, 30.0f, 30.0f);  
  position.x = Animate(0.0f, 25.0f, 5.0f);
  mShadowPass.projector.set_position(position);
  
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  render_shadow();
  blur_shadow();
  render_final(*mCamera);
}

void Application::init_scene() {
  aer::ShaderProxy &sp = aer::ShaderProxy::Get();
  sp.set_shader_path(DATA_DIRECTORY"/shaders/");
  sp.add_directive_token("*", "#version 420 core");
  sp.add_directive_token("CS", "#extension GL_ARB_compute_shader : enable");

  //---

  mProgram.shadow.create();
    mProgram.shadow.add_shader(sp.get("Shadowmap.create.VS"));
    mProgram.shadow.add_shader(sp.get("Shadowmap.create.FS"));
  AER_CHECK(mProgram.shadow.link());

  // Compute Shader kernel to blur input image
  AER_CHECK(mProgram.blur.create(sp.get("Blur_rg16f.CS")));

  mProgram.scene.create();
    mProgram.scene.add_shader(sp.get("Shadowmap.render.VS"));
    mProgram.scene.add_shader(sp.get("Shadowmap.render.FS"));
    mProgram.scene.add_shader(sp.get("Shadowmap.render.include", GL_FRAGMENT_SHADER));
  AER_CHECK(mProgram.scene.link());


  mScene.floorPlane.init(256.0f, 256.0f, 48u);

  mScene.sphere.init(5.0f, 32u);
  mScene.sphereModel = glm::translate(glm::mat4(1.0f), aer::Vector3(-5.0f, 7.0f, -10.0f));

  mScene.cube.init(5.0f);
  mScene.cubeModel = glm::translate(glm::mat4(1.0f), aer::Vector3(3.0f, 12.0f, -1.0f));
}

void Application::init_shadow() {
  const aer::U32 texres = 512u;

  // projection
  aer::View view(aer::Vector3(10.0f, 30.0f, 20.0f),
                 aer::Vector3(0.0f, 0.0f, 0.0f),
                 aer::Vector3(0.0f, 0.0f, 1.0f));

  mShadowPass.projector.set_view(view);
  mShadowPass.projector.set_frustum(aer::Frustum(90.0f, 1.0f, 0.1f, 1000.0f));

  // Raw shadowmap
  mShadowPass.texSHADOW.generate();
  mShadowPass.texSHADOW.bind();
  mShadowPass.texSHADOW.allocate(GL_RG16F, texres, texres);

  // Depth buffer used by the FBO
  mShadowPass.texDEPTH.generate();
  mShadowPass.texDEPTH.bind();
  mShadowPass.texDEPTH.allocate(GL_DEPTH_COMPONENT24, texres, texres);

  // Blurred shadowmap
  mShadowPass.texBLUR.generate();
  mShadowPass.texBLUR.bind();
  mShadowPass.texBLUR.allocate(GL_RG16F, texres, texres);
  //glGenerateMipmap(GL_TEXTURE_2D);
  aer::Texture::Unbind(GL_TEXTURE_2D);

  // Blurred shadow sampler
  mShadowPass.sampler.generate();

#if 0
  mShadowPass.sampler.set_wraps(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
#else
  mShadowPass.sampler.set_wraps(GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER);
  mShadowPass.sampler.set_border_color(aer::Vector4(255.0f));
#endif

#if 0
  /// If we use mipmapping
  mShadowPass.sampler.set_mag_filter(GL_LINEAR);
  mShadowPass.sampler.set_mipmap_min_filter(GL_LINEAR, GL_LINEAR);
#else
  mShadowPass.sampler.set_filters(GL_LINEAR, GL_LINEAR);
#endif
  mShadowPass.sampler.set_anisotropy_level(1.0f);

  /// VSM are not standard shadow map tested during texture lookup
  /// so we don't need to specify comparison states.
  //mShadowPass.sampler.set_compare_mode(GL_COMPARE_REF_TO_TEXTURE);
  //mShadowPass.sampler.set_compare_func(GL_LESS);

  // fbo
  mShadowPass.FBO.generate();
  mShadowPass.FBO.bind();
  mShadowPass.FBO.attach_color(&mShadowPass.texSHADOW, GL_COLOR_ATTACHMENT0);
  mShadowPass.FBO.attach_special(&mShadowPass.texDEPTH, GL_DEPTH_ATTACHMENT);
  AER_CHECK(aer::Framebuffer::CheckStatus());
  mShadowPass.FBO.unbind();

  CHECKGLERROR();  
}


void Application::draw_scene(const aer::Camera &camera, aer::Program &pgm) {
  pgm.set_uniform("uViewMatrix", camera.view_matrix());
  pgm.set_uniform("uViewProjectionMatrix", camera.view_projection_matrix());

  pgm.set_uniform("uModelMatrix", aer::Matrix4x4(1.0f));
  pgm.set_uniform("uColor", aer::Vector3(0.75f));
  mScene.floorPlane.draw();

  pgm.set_uniform("uModelMatrix", mScene.sphereModel);
  pgm.set_uniform("uColor", aer::Vector3(1.0f,0.0f,0.0f));
  mScene.sphere.draw();

  pgm.set_uniform("uModelMatrix", mScene.cubeModel);
  pgm.set_uniform("uColor", aer::Vector3(0.0f,0.0f,1.0f));
  mScene.cube.draw();
}


void Application::render_shadow() {
  aer::opengl::StatesInfo states = aer::opengl::PopStates();

  aer::U32 res = mShadowPass.texSHADOW.storage_info().width;
  glViewport(0, 0, res, res);
  
  //glEnable(GL_POLYGON_OFFSET_FILL);
  //glPolygonOffset(2.5f, 10.0f);
  
#if 1
  /// Original algorithm states to keep shadow casters
  glDisable(GL_CULL_FACE);
#else
  /// but we could try without afterwards
#endif

  mShadowPass.FBO.bind(GL_DRAW_FRAMEBUFFER);
  {
    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //---------------
    // The scene should be rendered with a simple passthough
    // (eg. set_scene_lod(shadowmap_lod))

    aer::Program &pgm = mProgram.shadow;
    pgm.activate();
    draw_scene(mShadowPass.projector, pgm);
    pgm.deactivate();
    //---------------
  }
  mShadowPass.FBO.unbind();

  aer::opengl::PushStates(states);

  CHECKGLERROR();
}

void Application::blur_shadow() {
  aer::Program &pgm = mProgram.blur;
  pgm.activate();
  {
    // Bind source texture in TEXTURE_UNIT 0
    aer::DefaultSampler::NearestClampled().bind(0u);
    mShadowPass.texSHADOW.bind(0u);
    pgm.set_uniform("uSrcTex", 0);

    // Bind destination texture to IMAGE_UNIT 0
    mShadowPass.texBLUR.bind_image(0u, GL_WRITE_ONLY);
    pgm.set_uniform("uDstImg", 0);

    // Launch kernel
    const aer::U32 kBlockDim = 16u;
    aer::U32 res = mShadowPass.texSHADOW.storage_info().width;
    aer::U32 gridDim = (res + kBlockDim-1u) / kBlockDim;
    glDispatchCompute(gridDim, gridDim, 1);

    // Unbind src & dst textures
    mShadowPass.texSHADOW.unbind();
    mShadowPass.texBLUR.unbind_image(0u);
    aer::DefaultSampler::NearestClampled().unbind(0u);
  }
  pgm.deactivate();

  CHECKGLERROR();
}

void Application::render_final(const aer::Camera &camera) {
  aer::Program &pgm = mProgram.scene;

  pgm.activate();
  {
    // note : bias is applied inside the program shader
    
    const aer::Matrix4x4& shadowMatrix = mShadowPass.projector.view_matrix();
    pgm.set_uniform("uShadowMatrix", shadowMatrix);

    const aer::Matrix4x4& shadowProjMatrix = mShadowPass.projector.view_projection_matrix();
    pgm.set_uniform("uShadowProjMatrix", shadowProjMatrix);

    mShadowPass.sampler.bind(0u);
#if 1
    mShadowPass.texBLUR.bind(0u);
#else
    mShadowPass.texSHADOW.bind(0u);
#endif
    pgm.set_uniform("uShadowMap", 0);

    draw_scene(camera, pgm);

    mShadowPass.sampler.unbind(0u);
    mShadowPass.texBLUR.unbind();
  }
  pgm.deactivate();

  CHECKGLERROR();
}
