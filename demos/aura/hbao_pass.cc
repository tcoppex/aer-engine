// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------


#include "aura/hbao_pass.h"

#include "aer/view/frustum.h"
#include "aer/rendering/mapscreen.h"



HBAOPass::HBAOPass() : mInputDepthTex(nullptr) {
  memset(&mUniformParams, 0, sizeof(mUniformParams));
}

HBAOPass::~HBAOPass() {
  deinit();
}

void HBAOPass::init(aer::Texture2D *hardware_depth_texture) {
  AER_ASSERT(nullptr != hardware_depth_texture);

  if (mInputDepthTex == hardware_depth_texture) {
    return;
  }

  mInputDepthTex = hardware_depth_texture;

  init_textures();
  init_shaders();
}

void HBAOPass::deinit() {
  if (nullptr == mInputDepthTex) {
    return;
  }

  mFBO.release();
  
  mTex.linDepth.release();
  mTex.AOX.release();
  mTex.AOXY.release();

  mProgram.linDepth.release();
  mProgram.ssao.release();
  mProgram.blurX.release();
  mProgram.blurY.release();

  mInputDepthTex = nullptr;
}

void HBAOPass::process(aer::Texture2D **output_ao_texture_pptr, 
                       const aer::Frustum& frustum) {
  AER_ASSERT(nullptr != mInputDepthTex);


  update_parameters(frustum);

  mFBO.bind(GL_DRAW_FRAMEBUFFER);
    glClear(GL_COLOR_BUFFER_BIT);
    linearize_depth();
    launch_kernel_HBAO();
    launch_kernel_blurAO();
    //compositing();
  mFBO.unbind();

  // deactivate any program let active
  aer::Program::Deactivate();
  aer::Sampler::UnbindAll(2u);

  *output_ao_texture_pptr = &mTex.blurAOXY;

  CHECKGLERROR();
}



void HBAOPass::init_textures() {
  const aer::U32 width  = mInputDepthTex->storage_info().width;
  const aer::U32 height = mInputDepthTex->storage_info().height;


  // - TEXTURES
  mTex.linDepth.generate();
  mTex.linDepth.bind();
  mTex.linDepth.allocate(GL_R32F, width, height);

  mTex.AOX.generate();
  mTex.AOX.bind();
  mTex.AOX.allocate(GL_R32F, width, height);

  mTex.AOXY.generate();
  mTex.AOXY.bind();
  mTex.AOXY.allocate(GL_RG16F, width, height);

  mTex.blurAOX.generate();
  mTex.blurAOX.bind();
  mTex.blurAOX.allocate(GL_RG16F, width, height);

  mTex.blurAOXY.generate();
  mTex.blurAOXY.bind();
  mTex.blurAOXY.allocate(GL_R32F, width, height);

  mTex.finalAO.generate();
  mTex.finalAO.bind();
  mTex.finalAO.allocate(GL_R32F, width, height);

  aer::Texture::Unbind(GL_TEXTURE_2D);

  // - FRAMEBUFFER
  mFBO.generate();
  mFBO.bind(GL_FRAMEBUFFER);
    mFBO.attach_color(&mTex.linDepth, GL_COLOR_ATTACHMENT0);
    mFBO.attach_color(&mTex.finalAO,  GL_COLOR_ATTACHMENT1);
  AER_CHECK(aer::Framebuffer::CheckStatus());
  mFBO.unbind();

  CHECKGLERROR();
}

void HBAOPass::init_shaders() {
  aer::ShaderProxy &sp = aer::ShaderProxy::Get();

  /// Linearize Depth Buffer
  mProgram.linDepth.create();
    mProgram.linDepth.add_shader(sp.get("MapScreen.VS"));
    mProgram.linDepth.add_shader(sp.get("LinearizeDepth.NoMSAA.FS"));
  AER_CHECK(mProgram.linDepth.link());

  /// HBAO Compute Shader
  AER_CHECK(mProgram.ssao.create(sp.get("HBAO.CS")));

  // Defines the kernel block dimension for the blur
  char directive_buffer[32];
  sprintf(directive_buffer, "#define BLOCK_DIM\t%d", kBlurBlockDim);
  sp.add_directive_token("BlurAO", directive_buffer);

  /// Horizontal & vertical blur passes
  AER_CHECK(mProgram.blurX.create(sp.get("BlurAO.X.CS")));
  AER_CHECK(mProgram.blurY.create(sp.get("BlurAO.Y.CS")));

  CHECKGLERROR();
}

/// Set the uniform parameters 'name' to program pgm
#define SET_UNIFORM_PARAM(pgm, name) \
     pgm.set_uniform("u"#name, mUniformParams._##name)


void HBAOPass::linearize_depth() {
  mProgram.linDepth.activate();
  {
    aer::DefaultSampler::NearestClampled().bind(0u);
    mInputDepthTex->bind(0u);
    mProgram.linDepth.set_uniform("uDepthTex", 0);
    
    SET_UNIFORM_PARAM(mProgram.linDepth, LinA);
    SET_UNIFORM_PARAM(mProgram.linDepth, LinB);

    aer::MapScreen::Draw();

    aer::Texture::Unbind(GL_TEXTURE_2D, 0u);
    aer::Sampler::Unbind(0u);
  }

  CHECKGLERROR();
}

void HBAOPass::launch_kernel_HBAO() {
  const aer::U32 width  = mInputDepthTex->storage_info().width;
  const aer::U32 height = mInputDepthTex->storage_info().height;


  aer::Program &pgm = mProgram.ssao;

  pgm.activate();
  {
    // Get subroutine info
    GLint suLoc = glGetSubroutineUniformLocation(pgm.id(), GL_COMPUTE_SHADER, "suHBAO");
    AER_CHECK(suLoc == 0);
    GLuint ssaoX_id = glGetSubroutineIndex(pgm.id(), GL_COMPUTE_SHADER, "HBAO_X");
    GLuint ssaoY_id = glGetSubroutineIndex(pgm.id(), GL_COMPUTE_SHADER, "HBAO_Y");

    //---

    /// X/Y-HBAO Inputs
    SET_UNIFORM_PARAM(pgm, AOResolution);
    SET_UNIFORM_PARAM(pgm, InvAOResolution);
    SET_UNIFORM_PARAM(pgm, UVToViewA);
    SET_UNIFORM_PARAM(pgm, UVToViewB);
    SET_UNIFORM_PARAM(pgm, R2);
    SET_UNIFORM_PARAM(pgm, TanAngleBias);
    SET_UNIFORM_PARAM(pgm, Strength);

    aer::I32 image_unit   = 0;
    aer::I32 texture_unit = 0;


    aer::DefaultSampler::NearestClampled().bind(0u);
    mTex.linDepth.bind(texture_unit);
    pgm.set_uniform("uTexLinDepth", texture_unit);
    ++texture_unit;

    aer::Vector2i gridDim;

    //--------

    // X-HBAO output
    mTex.AOX.bind_image(image_unit, GL_WRITE_ONLY);
    pgm.set_uniform("uImgOutputX", image_unit);
    ++image_unit;

    // Launch X-HBAO
    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &ssaoX_id);
    gridDim.x = (width + kHBAOTileWidth) / kHBAOTileWidth;
    gridDim.y = height;
    glDispatchCompute(gridDim.x, gridDim.y, 1);

CHECKGLERROR();
    //--------

    image_unit = 0;

    // Y-HBAO input
    mTex.AOX.bind_image(image_unit, GL_READ_ONLY);
    pgm.set_uniform("uImgAOX", image_unit);
    ++image_unit;

    // Y-HBAO output
    mTex.AOXY.bind_image(image_unit, GL_WRITE_ONLY);
    pgm.set_uniform("uImgOutputY", image_unit);
    ++image_unit;
    
    // Launch Y-HBAO
    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &ssaoY_id);
    gridDim.x = (height + kHBAOTileWidth) / kHBAOTileWidth;
    gridDim.y = width;
    glDispatchCompute(gridDim.x, gridDim.y, 1);
  }

  CHECKGLERROR();
}

void HBAOPass::launch_kernel_blurAO() {
  const aer::U32 width = mInputDepthTex->storage_info().width;
  const aer::U32 height = mInputDepthTex->storage_info().height;

  
  aer::DefaultSampler::NearestClampled().bind(0u);
  aer::DefaultSampler::LinearClampled().bind(1u);

  aer::Vector2i gridDim;

  /// Horizontal blur pass
  mProgram.blurX.activate();
  {
    aer::Program &pgm = mProgram.blurX;

    SET_UNIFORM_PARAM(pgm, BlurFalloff);
    SET_UNIFORM_PARAM(pgm, BlurDepthThreshold);
    SET_UNIFORM_PARAM(pgm, InvFullResolution);
    SET_UNIFORM_PARAM(pgm, FullResolution);

    // Input textures
    pgm.set_uniform("uTexAOLinDepthNearest", 0);
    pgm.set_uniform("uTexAOLinDepthLinear", 1);
    mTex.AOXY.bind(0);
    mTex.AOXY.bind(1);
    
    // Output image
    pgm.set_uniform("uDstImg", 0);
    mTex.blurAOX.bind_image(0, GL_WRITE_ONLY);

    /// Launch X-blur
    const aer::U32 kRowTileWidth = kHBAOTileWidth;
    gridDim.x = (width + kRowTileWidth) / kRowTileWidth;
    gridDim.y = height;
    glDispatchCompute(gridDim.x, gridDim.y, 1);
  }

  CHECKGLERROR();

  /// Vertical blur pass
  mProgram.blurY.activate();
  {
    aer::Program &pgm = mProgram.blurY;

    SET_UNIFORM_PARAM(pgm, BlurFalloff);
    SET_UNIFORM_PARAM(pgm, BlurDepthThreshold);
    SET_UNIFORM_PARAM(pgm, InvFullResolution);    
    SET_UNIFORM_PARAM(pgm, FullResolution);

    // Input textures
    pgm.set_uniform("uTexAOLinDepthNearest", 0);
    pgm.set_uniform("uTexAOLinDepthNearest", 1);
    mTex.blurAOX.bind(0);
    mTex.blurAOX.bind(1);

    // Output image
    pgm.set_uniform("uDstImg", 0);
    mTex.blurAOXY.bind_image(0, GL_WRITE_ONLY);

    /// Launch Y-Blur
    const aer::U32 kRowTileHeight = kHBAOTileWidth;
    gridDim.x = (height + kRowTileHeight) / kRowTileHeight;
    gridDim.y = width;
    glDispatchCompute(gridDim.x, gridDim.y, 1);
  }

  CHECKGLERROR();
}

void HBAOPass::compositing() {
  //
}

void HBAOPass::update_parameters(const aer::Frustum &frustum) {
  const aer::U32 width  = mInputDepthTex->storage_info().width;
  const aer::U32 height = mInputDepthTex->storage_info().height;

  const float zNear = frustum.znear();
  const float zFar  = frustum.zfar();

  const aer::Vector2 &lParams = frustum.linearization_params(); //

  const float sceneScale      = std::min(zNear, zFar);
  const float degToRad        = M_PI / 180.0f;
  const float fovyRad         = frustum.fov() * degToRad;
  const float radius          = 0.015f;
  const float radiusScale     = std::max(radius, 1.e-8f);
  const float blurRadius      = 16.0f;
  const float blurSigma       = (blurRadius + 1.0f) * 0.5f;
  const float invLn2          = 1.44269504f;
  const float sqrtLn2         = 0.832554611f;  
  const float blurSharpness   = 8.0f;


# define SETPARAM(p,v) mUniformParams._##p = v
# define GETPARAM(p)   mUniformParams._##p

  SETPARAM(FullResolution, aer::Vector2(width, height));
  SETPARAM(InvFullResolution, 1.0f / GETPARAM(FullResolution));

  SETPARAM(AOResolution, GETPARAM(FullResolution)); //
  SETPARAM(InvAOResolution, GETPARAM(InvFullResolution));//

  const aer::Vector2 &aor = GETPARAM(AOResolution);
  SETPARAM(FocalLen, 1.0f/tanf(fovyRad*0.5f) * aer::Vector2((aor.y/aor.x), 1.0f));
  SETPARAM(InvFocalLen, 1.0f / GETPARAM(FocalLen));

  SETPARAM(UVToViewA, GETPARAM(InvFocalLen)*aer::Vector2( 2.0f,-2.0f));
  SETPARAM(UVToViewB, GETPARAM(InvFocalLen)*aer::Vector2(-1.0f, 1.0f));

  SETPARAM(R, radiusScale*sceneScale);
  SETPARAM(R2, GETPARAM(R)*GETPARAM(R));
  SETPARAM(NegInvR2, -1.0f / GETPARAM(R2));
  SETPARAM(MaxRadiusPixels, 0.1f*std::min(width, height));

  SETPARAM(AngleBias, 10.0f * degToRad);
  SETPARAM(TanAngleBias, tanf(GETPARAM(AngleBias)));
  SETPARAM(PowExponent, 1.0f);
  SETPARAM(Strength,  1.0f);

  SETPARAM(BlurDepthThreshold, 2.0f * sqrtLn2 * (sceneScale / blurSharpness));
  SETPARAM(BlurFalloff, invLn2 / (2.0f*blurSigma*blurSigma));

  SETPARAM(LinA, zNear);//SETPARAM(LinA, lParams.x);
  SETPARAM(LinB, zFar);//SETPARAM(LinB, lParams.y);

# undef GETPARAM
# undef SETPARAM 
}