// -----------------------------------------------------------------------------
// Copyright 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef HAIR_HAIR_SIMULATION_H_
#define HAIR_HAIR_SIMULATION_H_

#include "aer/aer.h"
#include "aer/device/vertex_array.h"
#include "aer/device/device_buffer.h"
#include "aer/device/program.h"
#include "aer/device/texture_2d.h"
#include "aer/rendering/mesh.h"
#include "aer/physic/particle_system.h"
#include "aer/physic/verlet_integrator.h"

// =============================================================================

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
/// 
/// Actual implementation simulates the rendering of
/// one strand of hair, consisting of several control segment
/// tesselated & lerped between two ends.
/// 
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class HairSimulation {
 public:
  static const aer::U32 kNumInstances = 25u;
  static const aer::U32 kNumControlSegment = 4u;
  static const aer::U32 kNumControlPoints = kNumControlSegment + 1u;

  HairSimulation() :
    numlines_(16.0f),
    numlines_subsegment_(8.0f),
    bCurlHair_(true),
    bPauseSimulation_(false)
  {}

  void init();
  void update();
  void render(const aer::Camera &camera);

 private:
  void init_geometry();
  void init_textures();
  void init_shaders();
  void init_psystem();
  void init_devicebuffers();

  void events();
  void update_tangents();
  void update_devicebuffers();

  /// Simulation data
  aer::VerletIntegrator verlet_integrator_;
  aer::ParticleSystem_t particle_system_;

  // Base data
  aer::Mesh scalp_;
  GLuint    randtex_id_;

  // Rendering data
  aer::Mesh         control_hair_patch_;
  aer::Program      render_hair_pgm_;
  aer::Program      render_scalp_pgm_;
  std::vector<aer::Vector3> tangents_buffer_;

  /// Rendering parameters
  aer::F32 numlines_;
  aer::F32 numlines_subsegment_;
  
  /// States
  bool bCurlHair_;
  bool bPauseSimulation_;
};

// =============================================================================

#endif  // HAIR_HAIR_SIMULATION_H_