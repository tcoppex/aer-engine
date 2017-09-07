// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aer/physic/verlet_integrator.h"
#include "aer/physic/particle_system.h"


namespace aer {

void VerletIntegrator::simulate(const F32 dt,
                                const U32 nIteration,
                                ParticleSystem_t &psystem) {
  /// Resize the forces buffer if needed
  if (forces_accum_.empty() || 
      (psystem.NumParticles() < forces_accum_.size())) {
    forces_accum_.resize(psystem.NumParticles());
  }

  //AER_CHECK( psystem.NumForces() > 0u );
  //AER_CHECK( psystem.NumSprings() > 0u );

  /// Iterate the integrator
  for (U32 i = 0u; i < nIteration; ++i) {
    accumulateForces(dt, psystem);
    accumulateSprings(dt, psystem);
    integrate(dt, psystem.particles);
    satisfyConstraints(dt, psystem);
  }
}

void VerletIntegrator::accumulateForces(const F32 dt, 
                                        const ParticleSystem_t &psystem) {

  const ParticleBuffer_t &particles = psystem.particles;
  const float inv_dt = 1.0f / dt;

  /// Reset the forces buffer
  forces_accum_.assign(forces_accum_.size(), Vector3(0.0f));

  /// Accumulate forces
  for (U32 i = 0u; i < particles.Size(); ++i) {
    auto &forceAccum = forces_accum_[i];
   
    for (U32 j = 0u; j < psystem.NumForces(); ++j) {
      forceAccum += psystem.directional_forces[j];
    }
   
    /// Add Damping (viscosity)
    Vector3 vDamping = (particles.p1[i] - particles.p0[i]);
    forceAccum += psystem.params.global_damping * inv_dt * vDamping; 
  }
}

void VerletIntegrator::accumulateSprings(const F32 dt,
                                         const ParticleSystem_t &psystem) {
  const ParticleBuffer_t &particles = psystem.particles;

  for (U32 i = 0u; i < psystem.springs.size(); ++i) {
    const auto &spring = psystem.springs[i];
    
    const auto &ptA_last = particles.p0[spring.pointA];
    const auto &ptB_last = particles.p0[spring.pointB];
    const auto &ptA      = particles.p1[spring.pointA];
    const auto &ptB      = particles.p1[spring.pointB];

#if AER_DEBUG
    if (ptA == ptB) {
      AER_CHECK( ptA != ptB );
      printf("spring %d [%d, %d] : %f %f %f\n", i, spring.pointA, spring.pointB,
                                               ptA.x, ptA.y, ptA.z);
      break;
    }
#endif
    
    // velocity vectors
    Vector3 vA = (ptA - ptA_last) / dt;
    Vector3 vB = (ptB - ptB_last) / dt;

    // position & velocity difference
    Vector3 dp = ptA - ptB;
    Vector3 dv = vA - vB;

    // factors
    F32 dp_length = glm::length(dp);
    F32 shear_factor = -spring.Ks * (dp_length - spring.restLength);
    F32 damp_factor  = +spring.Kd * glm::dot( dv, dp) / dp_length;

    // final force
    dp = glm::normalize(dp);
    Vector3 spring_force = (shear_factor + damp_factor) * dp;

    forces_accum_[spring.pointA] += spring_force;
    forces_accum_[spring.pointB] -= spring_force;
  }
}

void VerletIntegrator::integrate(const F32 dt,
                                 ParticleBuffer_t &particles) {
  F32 dt2 = dt * dt;

  for (U32 i = 0u; i < particles.Size(); ++i) {
    Vector3 &P0 = particles.p0[i];
    Vector3 &P1 = particles.p1[i];

    Vector3 lastP1 = P1;
    P1 = 2.0f*P1 - P0 + dt2 * forces_accum_[i];
    P0 = lastP1;
  }
}

void VerletIntegrator::satisfyConstraints(const F32 dt,
                                          ParticleSystem_t &psystem) {
        auto& particles = psystem.particles;
  const auto& bspheres  = psystem.bounding_spheres;
  
  for (U32 i = 0u; i < particles.Size(); ++i) {
    const auto &old_position = particles.p0[i];
          auto &position     = particles.p1[i];

    // Bounding sphere collision
    F32 particle_radius = particles.radius[i];
    for (const auto &sphere : bspheres)
    {
      Vector3 sphere_to_particle = position - sphere.center;
      F32 d2 = glm::dot( sphere_to_particle, sphere_to_particle);
      F32 r = sphere.radius + particle_radius;

      if (d2 < r*r) {
        sphere_to_particle = r * glm::normalize(sphere_to_particle);
        position = sphere.center + sphere_to_particle;
      }
    }

#if AER_DEBUG
    // debug simulation bounding box
    position = glm::clamp(position, Vector3(-200.0f), Vector3(+200.0f));
#endif

    // Fixed particles
    if (particles.tied[i]) {
      position = old_position;
    }
  }
}

}  // namespace aer
