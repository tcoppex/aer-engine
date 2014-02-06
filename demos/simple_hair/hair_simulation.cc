// -----------------------------------------------------------------------------
// Copyright 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "simple_hair/hair_simulation.h"

#include <vector>
#include "glm/gtc/noise.hpp"
#include "aer/view/camera.h"


void HairSimulation::init() {
  init_psystem();
  init_devicebuffers();
  init_shaders();
}

void HairSimulation::update() {
  /// Update the particle system state
  const aer::U32 kNumIteration = 32u;
  aer::F32 dt = aer::GlobalClock::Get().delta_time(aer::SECOND);
  dt = std::max(dt, 0.5f/kNumIteration); //
  
  verlet_integrator_.simulate(dt, kNumIteration, particle_system_);

  /// Upload datas to GPU with updated tangents
  update_devicebuffers();

  /// Events
  const aer::EventsHandler &ev = aer::EventsHandler::Get();
  aer::F32 speed;

  speed = 0.5f;
  if (ev.key_down(aer::Keyboard::Left)) {
    numlines_ = (numlines_ > 1.0f) ? numlines_ - speed : 1.0f;
  }
  if (ev.key_down(aer::Keyboard::Right)) {
    numlines_ = (numlines_ < 32.0f) ? numlines_ + speed : 32.0f; 
  }

  speed = 0.25f;
  if (ev.key_down(aer::Keyboard::Down)) {
    numlines_subsegment_ = (numlines_subsegment_ > 1.0f) ? numlines_subsegment_ - speed : 1.0f;
  }
  if (ev.key_down(aer::Keyboard::Up)) {
    numlines_subsegment_ = (numlines_subsegment_ < 32.0f) ? numlines_subsegment_ + speed : 32.0f;
  }

  if (ev.key_pressed(aer::Keyboard::C)) {
    bCurlHair = !bCurlHair;
  }
}

void HairSimulation::render(const aer::Camera &camera) {
  //glPointSize(2);
  program_.activate();
    const aer::Matrix4x4 &mvp = camera.view_projection_matrix();
    program_.set_uniform("uMVP", mvp);

    // tesselation params
    program_.set_uniform("uNumLines",           aer::I32(numlines_));
    program_.set_uniform("uNumLinesSubSegment", aer::I32(numlines_subsegment_));

    glPatchParameteri(GL_PATCH_VERTICES, 4);
    mesh_.draw();

  program_.deactivate();

  CHECKGLERROR();
}

void HairSimulation::init_psystem() {
  numvertices_ = 2u*(kNumControlSegment + 1u);
  
  // 
  tangents_buffer_.resize(numvertices_);

  //
  aer::ParticleBuffer_t &pbuffer = particle_system_.particles;
  pbuffer.Resize(numvertices_);


  /// Particles --------
  const float w = 2.0f;
  const float start_h = 2.5f;
  for (aer::U32 i = 0u; i < numvertices_/2u; ++i) {
    // "hair" length grow exponentially (!! it make the system more unstable)
    float h = i*i * start_h; //i*start_h;
    pbuffer.p1[2*i+0] = aer::Vector3(-w, h, 0.0f);
    pbuffer.p1[2*i+1] = aer::Vector3(+w, h, 0.0f);

    // particle size used for collision detection
    pbuffer.radius[2*i+0] =
    pbuffer.radius[2*i+1] = 0.5f * (i / numvertices_) * h;
  }

  // copy current position to retarded position
  pbuffer.p0.assign(pbuffer.p1.begin(), pbuffer.p1.end());

  // tied the bottom particles
  pbuffer.tied[0u] = true;
  pbuffer.tied[1u] = true;
  /// -----------------


  /// Springs ---------
  auto& Springs = particle_system_.springs;
  const aer::U32 numsprings = 3u*(kNumControlSegment) + 2u*(kNumControlSegment-1u);
  Springs.resize(numsprings);
  
  // As always with simulation parameters, this prove right 
  // in certain condition, but could have to be changed in others.
  const aer::F32 Ks_struct = +150.0f;      // ~
  const aer::F32 Kd_struct = -10.0f;       // ~
  const aer::F32 Ks_bend   = +50.0f;       // ~
  const aer::F32 Kd_bend   = -5.0f;        // ~

  // Structural springs
  aer::U32 spring_idx = 0u;
  for (aer::U32 i = 0u; i <kNumControlSegment; ++i) {
    auto &s1 = Springs[spring_idx++];
    auto &s2 = Springs[spring_idx++];
    auto &s3 = Springs[spring_idx++];

    s1.type       = aer::Spring_t::STRUCTURAL;
    s1.pointA     = 2*i;
    s1.pointB     = 2*(i+1);
    s1.restLength = glm::length(pbuffer.p1[s1.pointB] - pbuffer.p1[s1.pointA]);
    s1.Ks         = Ks_struct;
    s1.Kd         = Kd_struct;

    s2.type       = aer::Spring_t::STRUCTURAL;
    s2.pointA     = s1.pointA + 1;
    s2.pointB     = s1.pointB + 1;
    s2.restLength = glm::length(pbuffer.p1[s2.pointB] - pbuffer.p1[s2.pointA]);
    s2.Ks         = Ks_struct;
    s2.Kd         = Kd_struct;

    // the two end of a strands are connected by a string to avoid false
    // interpolation when collision occured with a sphere
    s3.type       = aer::Spring_t::STRUCTURAL;
    s3.pointA     = s1.pointA;
    s3.pointB     = s2.pointA;
    s3.restLength = glm::length(pbuffer.p1[s3.pointB] - pbuffer.p1[s3.pointA]);
    s3.Ks         = Ks_struct;
    s3.Kd         = Kd_struct;
  }

  // Bend springs
  for (aer::U32 i = 0u; i <kNumControlSegment-1u; ++i) {
    auto &s1 = Springs[spring_idx++];
    auto &s2 = Springs[spring_idx++];

    s1.type       = aer::Spring_t::BEND;
    s1.pointA     = 2*i;
    s1.pointB     = 2*(i+2);
    s1.restLength = glm::length(pbuffer.p1[s1.pointB] - pbuffer.p1[s1.pointA]);
    s1.Ks         = Ks_bend;
    s1.Kd         = Kd_bend;

    s2.type       = aer::Spring_t::BEND;
    s2.pointA     = s1.pointA + 1;
    s2.pointB     = s1.pointB + 1;
    s2.restLength = glm::length(pbuffer.p1[s2.pointB] - pbuffer.p1[s2.pointA]);
    s2.Ks         = Ks_bend;
    s2.Kd         = Kd_bend;
  }
  /// -----------------

  /// Forces ----------
  aer::Vector3 force;

  force = 0.05f*aer::Vector3(0.0f, -9.81f, 0.0f); // gravity
  particle_system_.directional_forces.push_back(force);
  
  force = 0.005f*aer::Vector3(1.0f, 0.0f, 1.0f); // "wind"
  particle_system_.directional_forces.push_back(force);
  /// -----------------

  /// Bounding sphere collider
  aer::BoundingSphere_t bsphere;
  // 'head'
  bsphere.radius = 4.0f;
  bsphere.center = aer::Vector3(0.0f, -bsphere.radius, 0.0f);
  particle_system_.bounding_spheres.push_back(bsphere);
  /// -----------------

  /// Params ----------
  auto &params = particle_system_.params;
  params.global_damping = -0.10f;
  /// -----------------
}

void HairSimulation::init_devicebuffers() {
  /// Elements
  aer::U32 nelems = 4u*kNumControlSegment;
  std::vector<aer::U32> elements(nelems);
  aer::UPTR elements_bytesize = elements.size()*sizeof(elements[0]);

  for (aer::U32 i=0u; i<kNumControlSegment; ++i) {
    aer::U32 idx = 4u*i;
    aer::U32 base = (idx==0u) ? 0u : elements[idx-2u];
    elements[idx+0u] = base;
    elements[idx+1u] = base+1u;
    elements[idx+2u] = base+2u;
    elements[idx+3u] = base+3u;
  }

  /// Mesh creation
  mesh_.init(1u, true);
  mesh_.set_index_count(elements.size());
  mesh_.set_indices_type(GL_UNSIGNED_INT);
  mesh_.set_primitive_mode(GL_PATCHES);

  mesh_.begin_update();
  // Setup vertices info (datas are stored after the simulation update)
  aer::DeviceBuffer &vbo = mesh_.vbo();
  vbo.bind(GL_ARRAY_BUFFER);
  {
    const auto &Positions = particle_system_.particles.p1;
    const auto &Tangents  = tangents_buffer_;
    aer::UPTR offset = 0;
    aer::U32 bind = 0;

    // Positions   
    glBindVertexBuffer(bind, vbo.id(), offset, sizeof(Positions[0]));
    glVertexAttribFormat(bind, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribBinding(bind, bind);
    glEnableVertexAttribArray(bind);
    offset += Positions.size() * sizeof(Positions[0]);
    ++bind;

    // Tangents
    glBindVertexBuffer(bind, vbo.id(), offset, sizeof(Tangents[0]));
    glVertexAttribFormat(bind, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribBinding(bind, bind);
    glEnableVertexAttribArray(bind);
    //offset += Tangents.size() * sizeof(Tangents[0]);
    //++bind;
  }
  vbo.unbind();

  // Setup elements infos & datas
  aer::DeviceBuffer &ibo = mesh_.ibo();
  ibo.bind(GL_ELEMENT_ARRAY_BUFFER);
  {
    ibo.allocate(elements_bytesize, GL_STATIC_READ);
    ibo.upload(0u, elements_bytesize, elements.data());  
  }
  mesh_.end_update();

  CHECKGLERROR();
}

void HairSimulation::init_shaders() {
  aer::ShaderProxy &sp = aer::ShaderProxy::Get();

  program_.create();
    program_.add_shader(sp.get("HairGen.VS"));
    program_.add_shader(sp.get("HairGen.TCS"));
    program_.add_shader(sp.get("HairGen.TES"));
    program_.add_shader(sp.get("HairGen.FS"));
  AER_CHECK(program_.link());

  CHECKGLERROR();
}

void HairSimulation::update_devicebuffers() {
        auto &Tangents  = tangents_buffer_;
  const auto &Positions = particle_system_.particles.p1;

  /// Calculate new tangents

  // Add a curly effect
  aer::Vector3 curly(0.0f);
  
  if (bCurlHair) {
    curly = -10.0f*aer::Vector3(cos(4.0f*M_PI), -0.1f, sin(2.7f*M_PI));
  }

  // outer tangents
  Tangents[0] = aer::Vector3(0.0f, 1.0f, 0.0f);
  Tangents[1] = aer::Vector3(0.0f, 1.0f, 0.0f);
  Tangents[numvertices_-2u] = aer::Vector3(0.0f, -1.0f, 0.0f)+curly;
  Tangents[numvertices_-1u] = aer::Vector3(0.0f, -1.0f, 0.0f)+curly;

  // inner tangents
  for (aer::U32 i = 1u; i < numvertices_/2u - 1u; ++i) {
    if (bCurlHair) {
      float di = 10.0f*i / static_cast<float>(numvertices_/2u-1u);
      curly = -3.0f*aer::Vector3(cos(di*M_PI), -0.1f, sin(di*M_PI));
    }

    Tangents[2*i+0] = 0.5f*(Positions[2*(i+1)]    - Positions[2*(i-1)])     + curly;
    Tangents[2*i+1] = 0.5f*(Positions[2*(i+1)+1u] - Positions[2*(i-1)+1u])  + curly;
  }


  /// Upload data to GPU
  const aer::UPTR positions_bytesize = Positions.size() * sizeof(Positions[0]);
  const aer::UPTR tangents_bytesize  = Tangents.size()  * sizeof(Tangents[0]);
  const aer::U32 buffersize = positions_bytesize + tangents_bytesize;

  aer::DeviceBuffer &vbo = mesh_.vbo();
  
  vbo.bind(GL_ARRAY_BUFFER);
  vbo.allocate(buffersize, GL_DYNAMIC_READ);
  vbo.upload(                 0, positions_bytesize, Positions.data());
  vbo.upload(positions_bytesize,  tangents_bytesize,  Tangents.data());
  vbo.unbind();
}
