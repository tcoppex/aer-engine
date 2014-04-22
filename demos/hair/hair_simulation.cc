// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "hair/hair_simulation.h"

#include <vector>
#include "glm/gtc/noise.hpp"
#include "aer/view/camera.h"

//==============================================================================

const aer::U32 HairSimulation::kNumInstances;
const aer::U32 HairSimulation::kNumControlSegment;
const aer::U32 HairSimulation::kNumControlPoints;

//------------------------------------------------------------------------------

void HairSimulation::init() {
  init_geometry();
  init_textures();
  init_shaders();
  init_psystem();
  init_devicebuffers();
}

//------------------------------------------------------------------------------

void HairSimulation::update() {  
  aer::GlobalClock &clock = aer::GlobalClock::Get();

  // Handles simulation events
  events();

  /// Update simulation
  if (!bPauseSimulation_) {
    /// Update the particle system state
    aer::F32 dt = clock.delta_time(aer::SECOND);
    const aer::U32 kNumIteration = 32u;
    verlet_integrator_.simulate(dt, kNumIteration, particle_system_);
  } else {
    clock.pause();
  }
  
  /// Calculate new tangents
  update_tangents();

  /// Upload datas to device
  update_devicebuffers();
}

//------------------------------------------------------------------------------

void HairSimulation::render(const aer::Camera &camera) {
  const aer::Matrix4x4 &mvp = camera.view_projection_matrix();
  
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  // Render the scalp & hair's control points
  {
    aer::Program &pgm = render_scalp_pgm_;

    pgm.activate();
      pgm.set_uniform("uMVP", mvp);
      pgm.set_uniform("uColor", aer::Vector3(0.0f, 1.0f, 0.0f));
      scalp_.draw();

      glPointSize(4.0f);
      pgm.set_uniform("uColor", aer::Vector3(1.0f, 0.5f, 0.0f));
      control_hair_patch_.set_primitive_mode(GL_POINTS);
      control_hair_patch_.draw();


    pgm.deactivate();
  }

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  // Render hair with tesselation & interpolation
  {
    aer::Program &pgm = render_hair_pgm_;

    pgm.activate();
      pgm.set_uniform("uMVP", mvp);

      // tesselation params
      pgm.set_uniform("uNumLines",           aer::I32(numlines_));
      pgm.set_uniform("uNumLinesSubSegment", aer::I32(numlines_subsegment_));

      glBindTexture(GL_TEXTURE_1D, randtex_id_);
      pgm.set_uniform("uTexRandom", 0);

      pgm.set_uniform("uNumInstances", int(kNumInstances));

      glPointSize(1.0f);
      glPatchParameteri(GL_PATCH_VERTICES, 6);
      control_hair_patch_.set_primitive_mode(GL_PATCHES);
      control_hair_patch_.draw_instances(kNumInstances);
    pgm.deactivate();
  }


  CHECKGLERROR();
}

//------------------------------------------------------------------------------

// ==================================================
// This part is only to test the algorithm in the first
// draft. It should be generated automatically when importing
// raw data.


// Vertices data : position + normal
aer::F32 g_scalp_verts[] = {
  -10.0f, 0.0f, -10.0f,    0.0f, 1.0f, 0.0f,
  +10.0f, 0.0f, -10.0f,    0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, +20.0f,    0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, -40.0f,    0.0f, 1.0f, 0.0f,
   30.0f, 0.0f,   0.0f,    0.0f, 1.0f, 0.0f,
};

// Faces data : vertex index
aer::U32 g_scalp_faces[][3] = {
  {0, 1, 2}, {0, 1, 3}, {1, 3, 4}, {1, 2, 4}
};

const aer::U32 kScalpNumFace  = 4u;
const aer::U32 kScalpNumVertex = 5u;

// ==================================================

//------------------------------------------------------------------------------

void HairSimulation::init_geometry() {
  /// Scalp creation
  aer::Mesh &mesh = scalp_;

  mesh.init(1u, true);
  //mesh.set_vertex_count(kScalpNumVertex);
  mesh.set_primitive_mode(GL_TRIANGLES);

  mesh.begin_update();
  aer::DeviceBuffer &vbo = mesh.vbo();
  vbo.bind(GL_ARRAY_BUFFER);
  {
    vbo.allocate(sizeof(g_scalp_verts), GL_STATIC_READ);
    vbo.upload(0, sizeof(g_scalp_verts), g_scalp_verts);

    aer::U32 stride  = 3u * sizeof(g_scalp_verts[0]);   
    glBindVertexBuffer(0, vbo.id(), 0, 2*stride);

    // Positions
    glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribBinding(0, 0);
    glEnableVertexAttribArray(0);

    // Normals
    glVertexAttribFormat(1, 3, GL_FLOAT, GL_FALSE, stride);
    glVertexAttribBinding(1, 0);
    glEnableVertexAttribArray(1);
  }
  vbo.unbind();


  aer::U32 nelems = 3u*kScalpNumFace;
  mesh.set_index_count(nelems);
  mesh.set_indices_type(GL_UNSIGNED_INT);

  // Setup elements infos & datas
  aer::DeviceBuffer &ibo = mesh.ibo();
  ibo.bind(GL_ELEMENT_ARRAY_BUFFER);
  {

    const aer::UPTR elements_bytesize = nelems * sizeof(aer::U32);
    ibo.allocate(elements_bytesize, GL_STATIC_READ);
    ibo.upload(0u, elements_bytesize, (aer::U32*)(*g_scalp_faces));  
  }
  mesh.end_update();

  CHECKGLERROR();
}

//------------------------------------------------------------------------------

void HairSimulation::init_textures() {
  /// Setup the random lookup texture

  const aer::U32 texsize = 512u;
  aer::U8 *data = new aer::U8[3*texsize];

  srand(time(NULL));
  for (aer::U32 i = 0u; i < texsize; ++i) {
    aer::F32 x = rand() / static_cast<aer::F32>(RAND_MAX);
    aer::F32 y = rand() / static_cast<aer::F32>(RAND_MAX);
    aer::F32 z = rand() / static_cast<aer::F32>(RAND_MAX);
    aer::F32 n = x+y+z;

    if (n==0.0f) {
      n = 1.0f;
      x = y = z = 1.0f / 3.f;
    }

    x /= n;
    y /= n;
    z /= n;
    data[3*i+0] = static_cast<aer::U8>(255 * x);
    data[3*i+1] = static_cast<aer::U8>(255 * y);
    data[3*i+2] = static_cast<aer::U8>(255 * z);
  }


  glGenTextures(1, &randtex_id_);
  glBindTexture(GL_TEXTURE_1D, randtex_id_);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB8, texsize, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
  glBindTexture(GL_TEXTURE_1D, 0u);

  delete [] data;
  CHECKGLERROR();
}

//------------------------------------------------------------------------------

void HairSimulation::init_shaders() {
  aer::ShaderProxy &sp = aer::ShaderProxy::Get();

  // Program rendering the scalp
  {
    aer::Program &pgm = render_scalp_pgm_;
    pgm.create();
      pgm.add_shader(sp.get("PassThrough.VS"));
      pgm.add_shader(sp.get("PassThrough.FS"));
    AER_CHECK(pgm.link());
  }

  // Program rendering hair after tesselation / interpolation
  {
    aer::Program &pgm = render_hair_pgm_;
    pgm.create();
      pgm.add_shader(sp.get("HairGen.VS"));
      pgm.add_shader(sp.get("HairGen.TCS"));
      pgm.add_shader(sp.get("HairGen.TES"));
      pgm.add_shader(sp.get("HairGen.FS"));
    AER_CHECK(pgm.link());
  }

  CHECKGLERROR();
}

//------------------------------------------------------------------------------

void HairSimulation::init_psystem() {
  aer::U32 kNumControlPoints = kNumControlSegment + 1u;
  aer::U32 npoints = kScalpNumVertex * kNumControlPoints;

  tangents_buffer_.resize(npoints);

  aer::ParticleBuffer_t &pbuffer = particle_system_.particles;
  pbuffer.Resize(npoints);


  // --------- Particles --------

  const float segment_offset_factor = 1.75f;
# define SEGMENT_OFFSET(x)  (pow(x,segment_offset_factor))
  const float max_length = 55.0f;
  const float max_offset = SEGMENT_OFFSET(kNumControlPoints-1u);
  const float scale_offset = max_length / max_offset;

  // in buffer, each vertex store its control points in a row
  for (aer::U32 j = 0u; j < kScalpNumVertex; ++j) {
    aer::F32 *p_scalp = &g_scalp_verts[6u*j];

    aer::Vector3   base(p_scalp[0], p_scalp[1], p_scalp[2]);
    aer::Vector3 normal(p_scalp[3], p_scalp[4], p_scalp[5]);

    for (aer::U32 i = 0u; i < kNumControlPoints; ++i) {
      aer::U32 idx = j*kNumControlPoints + i;

      // segment length grow exponentially (!! it make the system more unstable)
      float offset = scale_offset * SEGMENT_OFFSET(i);
      pbuffer.p1[idx] = base + offset * normal;

      // particle size used for collision detection
      pbuffer.radius[idx] = 0.5f * offset * static_cast<float>(i / kNumControlPoints);

      // tied the bottom particles
      if (i < 2u) {
        pbuffer.tied[idx] = true;
      }
    }
  }

  // Copy current positions to delayed's
  pbuffer.reset_delayed_positions();



  // --------- Springs ---------

  auto& Springs = particle_system_.springs;
  const aer::U32 springs_per_vertex = kNumControlSegment + (kNumControlSegment-1u);
  const aer::U32 numsprings = kScalpNumVertex * springs_per_vertex;
  Springs.resize(numsprings);
  
  // As always with simulation parameters, this prove right 
  // in certain condition, but could have to be changed in others.
  const aer::F32 Ks_struct = +15.0f;      // ~
  const aer::F32 Kd_struct = -1.0f;       // ~
  const aer::F32 Ks_bend   = +5.0f;       // ~
  const aer::F32 Kd_bend   = -5.0f;       // ~

  aer::U32 spring_idx = 0u;

  for (aer::U32 j = 0u; j < kScalpNumVertex; ++j) {    
    aer::U32 idx = j * kNumControlPoints;

    // Structural springs
    for (aer::U32 i = 0u; i < kNumControlSegment; ++i) {
      auto &s = Springs[spring_idx++];

      s.type       = aer::Spring_t::STRUCTURAL;
      s.pointA     = idx + i;
      s.pointB     = idx + i + 1u;
      s.restLength = glm::length(pbuffer.p1[s.pointB] - pbuffer.p1[s.pointA]);
      s.Ks         = Ks_struct;
      s.Kd         = Kd_struct;
    }

    // Bend springs
    for (aer::U32 i = 0u; i <kNumControlSegment-1u; ++i) {
      auto &s = Springs[spring_idx++];

      s.type       = aer::Spring_t::BEND;
      s.pointA     = idx + i;
      s.pointB     = idx + i + 2u;
      s.restLength = glm::length(pbuffer.p1[s.pointB] - pbuffer.p1[s.pointA]);
      s.Ks         = Ks_bend;
      s.Kd         = Kd_bend;
    }
  }


  // --------- Forces ----------

  aer::Vector3 force;

  force = 0.05f*aer::Vector3(0.0f, -9.81f, 0.0f); // gravity
  particle_system_.directional_forces.push_back(force);
  
  force = 0.005f*aer::Vector3(1.0f, 0.0f, 1.0f); // "wind"
  particle_system_.directional_forces.push_back(force);


  // --------- Bounding sphere collider ---------

  aer::BoundingSphere_t bsphere;
  bsphere.radius = 40.0f; // 'head'
  bsphere.center = aer::Vector3(0.0f, -bsphere.radius, 0.0f);
  particle_system_.bounding_spheres.push_back(bsphere);
  

  // --------- Parameters ----------

  auto &params = particle_system_.params;
  params.global_damping = -0.1f;
}

//------------------------------------------------------------------------------


void HairSimulation::init_devicebuffers() {
  /// Elements
  // for each triangles we use 2 control points per vertex (start and end position
  // of the segment)
  const aer::U32 nelems = 6u * kNumControlSegment * kScalpNumFace;
  std::vector<aer::U32> elements(nelems);

  aer::U32 idx = 0u;
  for (aer::U32 j = 0u; j < kScalpNumFace; ++j) {
    aer::U32 *face = g_scalp_faces[j];
    for (aer::U32 i = 0u; i < kNumControlSegment; ++i) {
      aer::U32 e;

      e = kNumControlPoints * face[0u] + i;
      elements[idx+0u] = e;
      elements[idx+1u] = e + 1u;

      e = kNumControlPoints * face[1u] + i;
      elements[idx+2u] = e;
      elements[idx+3u] = e + 1u;

      e = kNumControlPoints * face[2u] + i;
      elements[idx+4u] = e;
      elements[idx+5u] = e + 1u;

      idx += 6u;
    }
  }

  //--------------------------

  aer::Mesh &mesh = control_hair_patch_;

  /// Mesh creation
  mesh.init(1u, true);
  mesh.set_index_count(elements.size());
  mesh.set_indices_type(GL_UNSIGNED_INT); //
  mesh.set_primitive_mode(GL_PATCHES);

  mesh.begin_update();
  // Setup vertices info (datas are stored after the simulation update)
  aer::DeviceBuffer &vbo = mesh.vbo();
  vbo.bind(GL_ARRAY_BUFFER);
  {
    const auto &Positions = particle_system_.particles.p1;
    const auto &Tangents  = tangents_buffer_;
    aer::UPTR offset = 0u;
    aer::U32 bind    = 0u;

    // Positions   
    glBindVertexBuffer(bind, vbo.id(), offset, sizeof(Positions[0u]));
    glVertexAttribFormat(bind, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribBinding(bind, bind);
    glEnableVertexAttribArray(bind);
    offset += Positions.size() * sizeof(Positions[0u]);
    ++bind;

    // Tangents
    glBindVertexBuffer(bind, vbo.id(), offset, sizeof(Tangents[0u]));
    glVertexAttribFormat(bind, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribBinding(bind, bind);
    glEnableVertexAttribArray(bind);
    //offset += Tangents.size() * sizeof(Tangents[0]);
    //++bind;
  }
  vbo.unbind();

  // Setup elements infos & datas
  aer::DeviceBuffer &ibo = mesh.ibo();
  ibo.bind(GL_ELEMENT_ARRAY_BUFFER);
  {
    const aer::UPTR elements_bytesize = elements.size() * sizeof(elements[0u]);
    ibo.allocate(elements_bytesize, GL_STATIC_READ);
    ibo.upload(0u, elements_bytesize, elements.data());  
  }
  mesh.end_update();

  CHECKGLERROR();
}

//------------------------------------------------------------------------------

void HairSimulation::events() {
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
    bCurlHair_ = !bCurlHair_;
  }
  
  if (ev.key_pressed(aer::Keyboard::Space)) {
    bPauseSimulation_ = !bPauseSimulation_;
  }
}

//------------------------------------------------------------------------------

void HairSimulation::update_tangents() {
        auto &Tangents  = tangents_buffer_;
  const auto &Positions = particle_system_.particles.p1;


  /// Compute Tangents
  aer::Vector3 curly(0.0f);
  for (aer::U32 j = 0u; j < kScalpNumVertex; ++j) {
    aer::U32 start_idx = j * kNumControlPoints;
    aer::U32 end_idx   = start_idx + kNumControlPoints - 1u;
    aer::F32 dj = (j+1.f) / kScalpNumVertex;

    // CuRlY !!!
    if (bCurlHair_) {
      aer::Vector2 v(sin(3.0f*dj), cos(5.0f));
      aer::F32 n = 1.25f * glm::simplex(v);
      curly = aer::Vector3(cos(n*4.0f*M_PI), -0.71f*n, sin(n*2.7f*M_PI));
    }

    // outer tangents
    Tangents[start_idx] = aer::Vector3(0.0f, 1.0f, 0.0f);
    Tangents[end_idx]   = aer::Vector3(0.0f, -1.0f, 0.0f) + curly;

    // inner tangents
    for (aer::U32 i = start_idx+1u; i < end_idx; ++i) {

      // CuRlY !!!
      if (bCurlHair_) {
        float di = (end_idx - i) / static_cast<float>(kNumControlPoints - 2u);
              di *= 10.0f;
        
        aer::Vector2 v(sin(3.0f*dj), cos(5.0f*di));
        aer::F32 n = di*glm::simplex(aer::Vector2(v));
        curly = -7.0f*aer::Vector3(1.7f*cos(n*M_PI), -0.3f*n, 2.5f*sin(n*M_PI));
      }

      Tangents[i] = 0.5f*(Positions[i+1u] - Positions[i-1u]) + curly;
    }
  }
}

//------------------------------------------------------------------------------

void HairSimulation::update_devicebuffers() {
        auto &Tangents  = tangents_buffer_;
  const auto &Positions = particle_system_.particles.p1;

  /// Upload data to GPU
  const aer::UPTR positions_bytesize = Positions.size() * sizeof(Positions[0u]);
  const aer::UPTR tangents_bytesize  = Tangents.size()  * sizeof(Tangents[0u]);
  const aer::U32 buffersize = positions_bytesize + tangents_bytesize;

  aer::DeviceBuffer &vbo = control_hair_patch_.vbo();
  
  vbo.bind(GL_ARRAY_BUFFER);
  vbo.allocate(buffersize, GL_DYNAMIC_READ); //
  vbo.upload(                 0, positions_bytesize, Positions.data());
  vbo.upload(positions_bytesize,  tangents_bytesize,  Tangents.data());
  vbo.unbind();
}

//==============================================================================
