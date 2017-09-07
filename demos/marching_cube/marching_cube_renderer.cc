// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aer/view/camera.h"

#include "marching_cube/marching_cube_renderer.h"
#include "marching_cube/lut_array.inc" // [to change]


const     aer::U32 MarchingCubeRenderer::kChunkDim;
constexpr aer::F32 MarchingCubeRenderer::kInvChunkDim;
constexpr aer::F32 MarchingCubeRenderer::kChunkSize;
constexpr aer::F32 MarchingCubeRenderer::kVoxelSize;

const     aer::U32  MarchingCubeRenderer::kMargin;
const     aer::U32  MarchingCubeRenderer::kWindowDim;
constexpr aer::F32  MarchingCubeRenderer::kInvWindowDim;


void MarchingCubeRenderer::init() {
  AER_ASSERT(!bInitialized_);

  init_geometry();
  init_textures();
  init_buffers();
  init_shaders();

  bInitialized_ = true;
}

void MarchingCubeRenderer::deinit() {
  if (!bInitialized_) {
    return;
  }

  density_tex_.release();
  density_rt_.release();

  trilist_tf_.release();
  tbo_.case_to_numtri.buffer.release();
  tbo_.case_to_numtri.texture.release();
  tbo_.edge_connect.buffer.release();
  tbo_.edge_connect.texture.release();

  for (auto &tf : tf_stack_) {
    tf.release();
  }

  //
  program_.build_density.release();
  program_.disp_density.release();
  program_.trilist.release();
  program_.genvertices.release();
  program_.render_chunk.release();

  bInitialized_ = false;
}

void MarchingCubeRenderer::generate(const aer::Vector3 &grid_dimension) {
  AER_ASSERT(bInitialized_);

  grid_dim_ = grid_dimension;
  grid_.resize(grid_dim_.x * grid_dim_.y * grid_dim_.z);

  aer::opengl::StatesInfo gl_states = aer::opengl::PopStates();

  glDisable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);
  glViewport(0u, 0u, kTextureRes, kTextureRes);

  aer::U32 index = 0u;
  for (int i = 0; i < grid_dim_.x; ++i) {
    for (int j = 0; j < grid_dim_.y; ++j) {
      for (int k = 0; k < grid_dim_.z; ++k) {
        auto &chunk = grid_[index++];
        chunk.i = i;
        chunk.j = j;
        chunk.k = k;
        create_chunk(chunk);
      }
    }
  }

  aer::opengl::PushStates(gl_states);
}


void MarchingCubeRenderer::render(const aer::Camera &camera) {
  AER_ASSERT(bInitialized_);

  aer::Program &pgm = program_.render_chunk;
  pgm.activate();

  const aer::Matrix4x4 &viewproj = camera.view_projection_matrix();
  pgm.set_uniform("uModelViewProjMatrix", viewproj);

  //----------
  const aer::U32 stride = 6u * sizeof(aer::F32);
  aer::U32 attrib = 0u;
  aer::U32 binding = 0u;
  aer::UPTR offset = 0u;

  glVertexAttribFormat(attrib, 3, GL_FLOAT, GL_FALSE, offset);
  glVertexAttribBinding(attrib, binding);
  glEnableVertexAttribArray(attrib);
  attrib++;

  offset = 3u * sizeof(aer::F32);
  glVertexAttribFormat(attrib, 3, GL_FLOAT, GL_FALSE, offset);
  glVertexAttribBinding(attrib, binding);
  attrib++;

  glEnableVertexAttribArray(0u);
  glEnableVertexAttribArray(1u);
  //----------

  aer::U32 index = 0u;
  for (int i = 0; i < grid_dim_.x; ++i) {
    for (int j = 0; j < grid_dim_.y; ++j) {
      for (int k = 0; k < grid_dim_.z; ++k) {
        auto &chunk = grid_[index++];

        if (chunk.id < 0 || chunk.state != CHUNK_FILLED) {
          continue;
        }

        //----------
        aer::DeviceBuffer &vbo = debug_vbos_[chunk.id];
        glBindVertexBuffer(binding, vbo.id(), 0, stride);
        vbo.bind(GL_ARRAY_BUFFER);
        //----------

        //render
        tf_stack_[chunk.id].draw(GL_TRIANGLES); //
      }
    }
  }

  //----------
  glDisableVertexAttribArray(0u);
  glDisableVertexAttribArray(1u);
  aer::DeviceBuffer::Unbind(GL_ARRAY_BUFFER);
  //----------

  aer::Program::Deactivate();

  CHECKGLERROR();
}


void MarchingCubeRenderer::init_geometry() {
  /// Quad mesh used to generate the 3D Volume slices
  slice_mesh_.init(1u, false);

  // position + texcoord
  const aer::F32 vertices[] = {-1.0f, +1.0f,  0.0f, 1.0f,
                               -1.0f, -1.0f,  0.0f, 0.0f,
                               +1.0f, +1.0f,  1.0f, 1.0f,
                               +1.0f, -1.0f,  1.0f, 0.0f};

  /// Setup Device buffer
  aer::DeviceBuffer &vbo = slice_mesh_.vbo();  
  vbo.bind(GL_ARRAY_BUFFER);
    vbo.allocate(sizeof(vertices), GL_STATIC_DRAW);
    vbo.upload(0, sizeof(vertices), vertices);
  vbo.unbind();

  /// Set buffer attributes
  slice_mesh_.begin_update();
    aer::I32 binding = 0;
    GLsizei stride = 4u * sizeof(vertices[0]);
    glVertexAttribFormat(binding, 4u, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribBinding(binding, binding);
    glBindVertexBuffer(binding, vbo.id(), 0, stride);
    glEnableVertexAttribArray(binding);
    ++binding;
  slice_mesh_.end_update();

  /// Set mesh rendering info
  slice_mesh_.set_primitive_mode(GL_TRIANGLE_STRIP);
  slice_mesh_.set_vertex_count(4u);


  CHECKGLERROR();
}


void MarchingCubeRenderer::init_textures() {
  sampler_.nearest_clamp.generate();
  sampler_.nearest_clamp.set_filters(GL_NEAREST, GL_NEAREST);
  sampler_.nearest_clamp.set_wraps(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

  sampler_.linear_clamp.generate();
  sampler_.linear_clamp.set_filters(GL_LINEAR, GL_LINEAR);
  sampler_.linear_clamp.set_wraps(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

  /// Generate the Density Volume Texture
  density_tex_.generate();
  density_tex_.bind();
  density_tex_.allocate(GL_R32F, kTextureRes, kTextureRes, kTextureRes);
  density_tex_.unbind();

  /// Render Texture
  density_rt_.generate();
  density_rt_.bind(GL_FRAMEBUFFER);
  density_rt_.attach_color(&density_tex_, GL_COLOR_ATTACHMENT0);
  AER_CHECK(aer::Framebuffer::CheckStatus()); //
  aer::Framebuffer::Unbind();


  CHECKGLERROR();
}


void MarchingCubeRenderer::init_buffers() {
  aer::U32 bytesize = 0u;

  /// Buffer used to generate the triangle list.
  // (A 2D grid of points, instanced as 'depth' layers)
  // note : internal format coul be replaced for 'integer'
  trilist_base_buffer_.generate();
  trilist_base_buffer_.bind(GL_ARRAY_BUFFER);
  bytesize = 2u * kChunkDim * kChunkDim * sizeof(aer::F32);
  trilist_base_buffer_.allocate(bytesize, GL_STATIC_DRAW);

  aer::F32 *d_indices(nullptr);
  trilist_base_buffer_.map(&d_indices, GL_WRITE_ONLY);
  {
    aer::U32 index = 0u;
    for (aer::U32 j = 0u; j < kChunkDim; ++j) {
      for (aer::U32 i = 0u; i < kChunkDim; ++i) {
        d_indices[index++] = static_cast<aer::F32>(i);
        d_indices[index++] = static_cast<aer::F32>(j);
      }
    }
  }
  trilist_base_buffer_.unmap(&d_indices);

  /// Buffer containing the triangle list packed as integer.
  // (3x6bits for voxel coordinates, 3x4bits for edges indices)
  trilist_buffer_.generate();
  trilist_buffer_.bind(GL_ARRAY_BUFFER);
  bytesize = kMaxTrianglesPerVoxel * kChunkDim * kChunkDim * kChunkDim * sizeof(aer::I32);
  trilist_buffer_.allocate(bytesize, GL_DYNAMIC_DRAW); //

  aer::DeviceBuffer::Unbind(GL_ARRAY_BUFFER);  


  /// Transform feedback used to generate the triangle list
  trilist_tf_.generate();
  trilist_tf_.bind();
  //aer::TransformFeedback::BindBufferBase(offset, trilist_buffer_);
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0u, trilist_buffer_.id()); //
  aer::TransformFeedback::Unbind();


  /// Texture Buffers
  // Setup texture buffers for LUT storing :
  // The number of triangle per voxel case (256 bytes value in [0-5])
  {
    aer::TBO_t &tbo = tbo_.case_to_numtri;

    bytesize = sizeof(s_caseToNumpolys);
    tbo.buffer.generate();
    tbo.buffer.bind(GL_TEXTURE_BUFFER);
    tbo.buffer.allocate(bytesize, GL_STATIC_DRAW);
    tbo.buffer.upload(0, bytesize, s_caseToNumpolys);

    tbo.texture.generate();
    tbo.texture.bind();
    tbo.texture.set_buffer(GL_R8I, tbo.buffer);
  }

  // The 0 to 5 trio of edge indices where lies the triangle vertices 
  // (5*256 short value, stored as 3x4bits)
  {
    aer::TBO_t &tbo = tbo_.edge_connect;

    bytesize = sizeof(s_packedEdgesIndicesPerCase);
    tbo.buffer.generate();
    tbo.buffer.bind(GL_TEXTURE_BUFFER);    
    tbo.buffer.allocate(bytesize, GL_STATIC_DRAW);
    tbo.buffer.upload(0, bytesize, s_packedEdgesIndicesPerCase);

    tbo.texture.generate();
    tbo.texture.bind();
    tbo.texture.set_buffer(GL_R16I, tbo.buffer);
  }

  aer::DeviceBuffer::Unbind(GL_TEXTURE_BUFFER);
  aer::Texture::Unbind(GL_TEXTURE_BUFFER);

  
  // List of free buffers [tmp]
  freebuffer_indices_.resize(kBufferBatchSize);
  nfreebuffers_ = freebuffer_indices_.size();

  // [TODO]
  // The total space used is not explained for now (should be large enough).
  // We use a factor of 6 to have aligned offset when using
  // bindBufferRange [see glspec43 p398]
  aer::U32 subbuffer_bytesize = (64u * 1024u) * (6u * sizeof(aer::F32)); //

  // Transform feedback stacks
  tf_stack_.resize(kBufferBatchSize);
  for (aer::U32 i = 0u; i < kBufferBatchSize; ++i) {
    auto &tf = tf_stack_[i];

    aer::DeviceBuffer &vbo = debug_vbos_[i];
    vbo.generate();
    vbo.bind(GL_ARRAY_BUFFER);
    vbo.allocate(subbuffer_bytesize, GL_DYNAMIC_DRAW);

    tf.generate();
    tf.bind();
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0u, vbo.id());

    freebuffer_indices_[i] = i; //
  }
  aer::TransformFeedback::Unbind();


  CHECKGLERROR();
}


void MarchingCubeRenderer::init_shaders() {
  aer::ShaderProxy &sp = aer::ShaderProxy::Get();

  /// First build pass : generate density volume
  {
    aer::Program &pgm = program_.build_density;

    pgm.create();
      pgm.add_shader(sp.get("1_BuildDensityVolume.VS"));
      pgm.add_shader(sp.get("1_BuildDensityVolume.GS"));
      pgm.add_shader(sp.get("1_BuildDensityVolume.FS"));
      // Manually include external shader
      pgm.add_shader(sp.get("Noise.Include", GL_FRAGMENT_SHADER));
      pgm.add_shader(sp.get("Density.Include", GL_FRAGMENT_SHADER));
    AER_CHECK(pgm.link());
    
    srand(time(nullptr));
    aer::I32 seed = 4567891 * (rand() / aer::F32(RAND_MAX));
    AER_DEBUG_CODE(fprintf(stderr, "Noise seed used: %d\n", seed);)

    pgm.activate();
    pgm.set_uniform("uPermutationSeed", seed);
    aer::Program::Deactivate();
  }

  /// First build pass rendering [for debugging]
  {
    aer::Program &pgm = program_.disp_density;

    pgm.create();
      pgm.add_shader(sp.get("DispDensityVolume.VS"));
      pgm.add_shader(sp.get("DispDensityVolume.FS"));
    AER_CHECK(pgm.link());
  }

  /// Second build pass : list triangle to generate
  {
    aer::Program &pgm = program_.trilist;

    pgm.create();
    pgm.add_shader(sp.get("2_ListTriangle.VS"));
    pgm.add_shader(sp.get("2_ListTriangle.GS"));

    const GLchar* varyings[] = {"x6y6z6_e4e4e4"};
    pgm.transform_feedback_varyings(AER_ARRAYSIZE(varyings),
                                    varyings,
                                    GL_INTERLEAVED_ATTRIBS);

    AER_CHECK(pgm.link());
  }


  /// Third build pass : generate vertices buffers
  {
    aer::Program &pgm = program_.genvertices;

    pgm.create();
    pgm.add_shader(sp.get("3_GenVertices.VS"));
    pgm.add_shader(sp.get("3_GenVertices.GS"));

    const GLchar* varyings[] = {"outPositionWS", "outNormalWS"};
    pgm.transform_feedback_varyings(AER_ARRAYSIZE(varyings),
                                    varyings,
                                    GL_INTERLEAVED_ATTRIBS);

    AER_CHECK(pgm.link());
  }

  /// Rendering pass
  {
    aer::Program &pgm = program_.render_chunk;

    pgm.create();
      pgm.add_shader(sp.get("PassThrough.VS"));
      pgm.add_shader(sp.get("PassThrough.FS"));
    AER_CHECK(pgm.link());
  }

  CHECKGLERROR();
}


//-------------
//-------------


void MarchingCubeRenderer::create_chunk(ChunkInfo_t &chunk) {
  //---------------
  // TODO : move outside
  static GLuint sQuery = 0u;

  GLint query_result = 0;
  if (0u == sQuery) {
    glGenQueries(1, &sQuery);
  }
  //---------------

  if (chunk.id < 0) {
    aer::Vector3 start = - 0.5f * grid_dim_;
    aer::Vector3 grid_coords = aer::Vector3(chunk.i, chunk.j, chunk.k);

    chunk.ws_coords = kChunkSize * (start + grid_coords);
    chunk.id = -1;
    chunk.state = CHUNK_EMPTY;
  }

  /// 1) Generate the density volume
  build_density_volume(chunk);

  glEnable(GL_RASTERIZER_DISCARD);

  /// 2) List triangles to output
  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, sQuery);
    list_triangles();
  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
  glGetQueryObjectiv(sQuery, GL_QUERY_RESULT, &query_result);
  bool bHasTriangles = (query_result > 0);

  /// 3) Generate the chunk's triangles (if any)
  if (bHasTriangles) {
    generate_vertices(chunk);
    chunk.state = CHUNK_FILLED;
    //fprintf(stderr, "%d : %d triangles listed\n", chunk.id, query_result);
  }

  glDisable(GL_RASTERIZER_DISCARD);
}

void MarchingCubeRenderer::build_density_volume(ChunkInfo_t &chunk) {
  density_rt_.bind(GL_DRAW_FRAMEBUFFER);

  aer::Program &pgm = program_.build_density;
  pgm.activate();

  aer::F32 scale_coord = kTextureRes * kInvWindowDim;
  aer::F32 texel_size  = 1.0f / static_cast<aer::F32>(kTextureRes);

  pgm.set_uniform("uChunkPositionWS", chunk.ws_coords);
  pgm.set_uniform("uChunkSizeWS",     kChunkSize);
  pgm.set_uniform("uInvChunkDim",     kInvChunkDim);
  pgm.set_uniform("uScaleCoord",      scale_coord);
  pgm.set_uniform("uTexelSize",       texel_size);
  pgm.set_uniform("uMargin",          aer::F32(kMargin));
  pgm.set_uniform("uWindowDim",       aer::F32(kWindowDim));

  //float t = aer::GlobalClock::Get().application_time();
  //pgm.set_uniform("uTime",       t);

  /// Compute the density on each voxels' corners,
  /// thus rendering N+1 slices.
  slice_mesh_.draw_instances(kTextureRes);

  pgm.deactivate();
  density_rt_.unbind();

  CHECKGLERROR();
}

void MarchingCubeRenderer::list_triangles() {
  aer::Program &pgm = program_.trilist;

  pgm.activate();
  pgm.set_uniform("uMargin",      aer::F32(kMargin));

  // note : TEXTURE_BUFFER & TEXTURE_3D can also use the same unit
  aer::I32 texunit = 0;

  pgm.set_uniform("uDensityVolume_nearest", texunit);
  sampler_.nearest_clamp.bind(texunit);
  density_tex_.bind(texunit);
  ++texunit;

  pgm.set_uniform("uCaseToNumTri", texunit);
  tbo_.case_to_numtri.texture.bind(texunit);
  ++texunit;

  pgm.set_uniform("uEdgeConnectList", texunit);
  tbo_.edge_connect.texture.bind(texunit);
  ++texunit;


  trilist_tf_.bind();
  aer::TransformFeedback::begin(GL_POINTS);
  {
#if 1
    trilist_base_buffer_.bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0u);

    /// List triangles contains in every voxels
    glDrawArraysInstanced(GL_POINTS, 0, kVoxelsPerSlice, kChunkDim); //

    glDisableVertexAttribArray(0u);
    trilist_base_buffer_.unbind();  
#endif
  }
  aer::TransformFeedback::end();
  trilist_tf_.unbind();

  aer::Sampler::UnbindAll(texunit);
  aer::Texture::UnbindAll(GL_TEXTURE_3D, texunit);
  aer::Texture::UnbindAll(GL_TEXTURE_BUFFER, texunit);

  pgm.deactivate();

  CHECKGLERROR();
}

void MarchingCubeRenderer::generate_vertices(ChunkInfo_t &chunk) {
  if ((chunk.id < 0) && (nfreebuffers_ <= 0)) {
    AER_WARNING("No more free space for chunk datas");
    chunk.id = -1; //
    return;    
  }

  if (chunk.id < 0) {
    chunk.id = freebuffer_indices_[kBufferBatchSize - nfreebuffers_];
    --nfreebuffers_;
  }

  aer::TransformFeedback &tf = tf_stack_[chunk.id];
  aer::Program &pgm = program_.genvertices;

  pgm.activate();
  
  pgm.set_uniform("uChunkPositionWS",   chunk.ws_coords);
  pgm.set_uniform("uVoxelSize",         kVoxelSize); //
  pgm.set_uniform("uMargin",            aer::F32(kMargin)); //
  pgm.set_uniform("uInvWindowDim",      kInvWindowDim); //
  pgm.set_uniform("uWindowDim",         aer::F32(kWindowDim)); //

  aer::I32 texunit = 0;

  pgm.set_uniform("uDensityVolume_nearest", texunit);
  sampler_.nearest_clamp.bind(texunit);
  density_tex_.bind(texunit);
  ++texunit;

  pgm.set_uniform("uDensityVolume_linear", texunit);
  sampler_.linear_clamp.bind(texunit);
  density_tex_.bind(texunit);
  ++texunit;


  tf.bind();
  aer::TransformFeedback::begin(GL_POINTS);
  {
#if 1
    trilist_buffer_.bind(GL_ARRAY_BUFFER);
    // Values are stored as packed integer [IMPORTANT]
    glVertexAttribIPointer(0, 1, GL_INT, 0, nullptr);
    glEnableVertexAttribArray(0u);

    trilist_tf_.draw(GL_POINTS);

    glDisableVertexAttribArray(0u);
    trilist_buffer_.unbind();  
#endif
  }
  aer::TransformFeedback::end();
  tf.unbind();

  aer::Sampler::UnbindAll(texunit);
  aer::Texture::UnbindAll(GL_TEXTURE_3D, texunit);

  pgm.deactivate();

  CHECKGLERROR();
}
