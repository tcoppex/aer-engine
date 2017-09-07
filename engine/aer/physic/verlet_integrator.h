// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_PHYSIC_VERLET_INTEGRATOR_H_
#define AER_PHYSIC_VERLET_INTEGRATOR_H_

#include <vector>
#include "aer/common.h"

namespace aer {

struct ParticleBuffer_t;
struct ParticleSystem_t;

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// A simple CPU springs-particles Verlet integrator.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class VerletIntegrator {
 public:
  void simulate(const F32 dt, 
                const U32 nIteration,
                ParticleSystem_t &psystem);

 private:
    void accumulateForces(const F32 dt, 
                          const ParticleSystem_t &psystem);

    void accumulateSprings(const F32 dt, 
                           const ParticleSystem_t &psystem);

    void integrate(const F32 dt,
                   ParticleBuffer_t &particles);

    void satisfyConstraints(const F32 dt,
                            ParticleSystem_t &psystem);


    /// Shared buffer storing accumulated forces
    std::vector<Vector3> forces_accum_;
};

}  // namespace aer

#endif  // AER_PHYSIC_VERLET_INTEGRATOR_H_
