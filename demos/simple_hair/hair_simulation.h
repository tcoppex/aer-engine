// -----------------------------------------------------------------------------
// Copyright 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef SIMPLE_HAIR_HAIR_SIMULATION_H_
#define SIMPLE_HAIR_HAIR_SIMULATION_H_

#include "aer/aer.h"
#include "aer/device/vertex_array.h"
#include "aer/device/device_buffer.h"
#include "aer/device/program.h"
#include "aer/rendering/mesh.h"
#include "aer/physic/particle_system.h"
#include "aer/physic/verlet_integrator.h"


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
/// 
/// Actual implementation simulates the rendering of
/// one strand of hair, consisting of several control segment
/// tesselated & lerped between two ends.
/// 
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class HairSimulation {
 public:
  static const aer::U32 kNumControlSegment = 4u;

  HairSimulation() :
    numvertices_(0u),
    numlines_(16.0f),
    numlines_subsegment_(8.0f),
    bCurlHair(true)
  {}

  void init();
  void update();
  void render(const aer::Camera &camera);

 private:
  void init_psystem();
  void init_devicebuffers();
  void init_shaders();

  void update_devicebuffers();


  /// Simulation datas
  aer::VerletIntegrator verlet_integrator_;
  aer::ParticleSystem_t particle_system_;

  // Rendering datas
  aer::U32          numvertices_;
  aer::Mesh         mesh_;
  aer::Program      program_;
  std::vector<aer::Vector3> tangents_buffer_;

  /// Rendering parameters
  aer::F32 numlines_;
  aer::F32 numlines_subsegment_;
  bool bCurlHair;
};

#endif  // SIMPLE_HAIR_HAIR_SIMULATION_H_