// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

/*
Le lecteur qui aime à penser verra dans ces mémoires que n'ayant jamais visé à 
un point fixe, le seul système que j'eus, si c'en est un, fut celui de me laisser
aller où le vent qui soufflait me poussait.
*/

#ifndef AER_PHYSIC_PARTICLE_SYSTEM_H_
#define AER_PHYSIC_PARTICLE_SYSTEM_H_

#include <vector>
#include "aer/common.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Constraint between two particles
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
struct Spring_t {
  enum Type {
    STRUCTURAL,
    SHEAR,
    BEND,
    kNumSpringType
  } type;

  U32 pointA;
  U32 pointB;
  F32 restLength;
  F32 Ks;
  F32 Kd;
};

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Set of particle states buffers
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
struct ParticleBuffer_t {
  U32 Size() const {return p0.size();}

  void Resize(size_t size) {
    p0.resize(size);
    p1.resize(size);
    radius.resize(size, 0.0f);
    tied.resize(size, false);
  }

  void reset_delayed_positions() {
    p0.assign(p1.begin(), p1.end());
  }

  std::vector<Vector3> p0;        //< last position
  std::vector<Vector3> p1;        //< current position
  std::vector<F32>     radius;    //< radius use for collision
  std::vector<bool>    tied;      //< true if particle static [prefer inverse of mass, with 0 = fixed]
};

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Set of sphere to collide particles with
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
struct BoundingSphere_t {
  Vector3 center;
  F32 radius;
};

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Embed a set of particles with their associated forces
/// & constraints.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
struct ParticleSystem_t {
  struct Params_t {
    F32 global_damping;
  };

  U32 NumParticles() const {return particles.Size();}
  U32 NumSprings()   const {return springs.size();}
  U32 NumForces()    const {return directional_forces.size();}

  Params_t                      params;               // System's parameters
  ParticleBuffer_t              particles;            // Particles
  std::vector<Spring_t>         springs;              // Springs constraints
  std::vector<Vector3>          directional_forces;   // Directional forces
  std::vector<BoundingSphere_t> bounding_spheres;     // Bounding spheres
};

}  // namespace aer

#endif  // AER_PHYSIC_PARTICLE_SYSTEM_H_
