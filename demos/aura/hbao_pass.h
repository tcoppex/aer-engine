// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AURA_HBAO_PASS_H_
#define AURA_HBAO_PASS_H_

#include "aer/aer.h"
#include "aer/device/sampler.h"
#include "aer/device/texture_2d.h"
#include "aer/device/program.h"
#include "aer/device/framebuffer.h"


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// Horizontal Based Ambient Occlusion algorithm CPU pass.
///
/// Original algorithm from Louis Bavoil, Nvidia.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class HBAOPass {
 public:
  HBAOPass();
  ~HBAOPass();

  void init(aer::Texture2D *hardware_depth_texture);
  void deinit();

  void process(aer::Texture2D **output_ao_texture_pptr,
               const aer::Frustum& frustum);


 private:
  static const aer::U32 kHBAOTileWidth = 320u;
  static const aer::U32 kBlurRadius    = 8u;
  static const aer::U32 kBlurBlockDim  = kHBAOTileWidth + 2*kBlurRadius;

  void init_textures();
  void init_shaders();

  void linearize_depth();
  //void downsample_depth();
  void launch_kernel_HBAO();
  void launch_kernel_blurAO();
  void compositing(); //

  void update_parameters(const aer::Frustum &frustum);


  aer::Texture2D *mInputDepthTex;

  struct {
    aer::Sampler nearest;
    aer::Sampler linear;
  } mSampler;

  /// Input / Output buffers [TODO: compact]
  struct {
    aer::Texture2D linDepth;      // [r32f] output LinDepth, input HBAO-X/-Y
    aer::Texture2D AOX;           // [r32f] output HBAO-X, input HBAO-Y
    aer::Texture2D AOXY;          // [rg16f] output HBAO-Y, input blurAO-X
    aer::Texture2D blurAOX;       // [rg16f] output blurAO-X, input blurAO-Y
    aer::Texture2D blurAOXY;      // [rg32f] output blurAO-XY, input compose
    aer::Texture2D finalAO;       // [rg32f] output compose
  } mTex;

  aer::Framebuffer mFBO;

  /// Passes program shader
  struct {
    aer::Program linDepth;        // Fragment Shader
    aer::Program ssao;            // Compute Shader
    aer::Program blurX;           // Compute Shader
    aer::Program blurY;           // Compute Shader
  } mProgram;

  /// HBAO parameters to be sent as uniforms
  struct Params_t {
    aer::Vector2 _FullResolution;
    aer::Vector2 _InvFullResolution;

    aer::Vector2 _AOResolution;
    aer::Vector2 _InvAOResolution;

    aer::Vector2 _FocalLen;
    aer::Vector2 _InvFocalLen;

    aer::Vector2 _UVToViewA;
    aer::Vector2 _UVToViewB;

    aer::F32 _R;
    aer::F32 _R2;
    aer::F32 _NegInvR2;
    aer::F32 _MaxRadiusPixels;

    aer::F32 _AngleBias;
    aer::F32 _TanAngleBias;
    aer::F32 _PowExponent;
    aer::F32 _Strength;

    aer::F32 _BlurDepthThreshold;
    aer::F32 _BlurFalloff;
    aer::F32 _LinA;
    aer::F32 _LinB;
  } mUniformParams;


  //GPUTimer_t mGPUTimer;
};

#endif  // AURA_HBAO_PASS_H_
