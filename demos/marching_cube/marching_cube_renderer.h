// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef MARCHING_CUBE_RENDERER_H_
#define MARCHING_CUBE_RENDERER_H_

#include <vector>

#include "aer/aer.h"
//#include "aer/device/query.h"
#include "aer/device/device_buffer.h"
#include "aer/device/program.h"
#include "aer/device/transform_feedback.h"
#include "aer/device/sampler.h"
#include "aer/device/texture_3d.h"
#include "aer/device/texture_buffer.h"
#include "aer/device/framebuffer.h"
#include "aer/rendering/mesh.h"



/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// 
/// Original algorithm by Ryan Geiss
/// (cf http://http.developer.nvidia.com/GPUGems3/gpugems3_ch01.html)
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class MarchingCubeRenderer {
 public:
  // Number of voxel per dimension for a chunk
  static const     aer::U32 kChunkDim       = 32u;

  // Number of voxel in a slice
  static const     aer::U32 kVoxelsPerSlice = kChunkDim * kChunkDim;

  //---------------------

  // We need to use a marge for algorithm looking on border chunk.
  // (eg. normal calculation)
  static const     aer::U32 kMargin       = 1u;
  static const     aer::U32 kWindowDim    = kChunkDim + 2u * kMargin;
  static constexpr aer::F32 kInvWindowDim = 1.0f / static_cast<aer::F32>(kWindowDim);

  // Density volume resolution (# of voxel corners)
  static const     aer::U32 kTextureRes   = kWindowDim + 1u;

  //---------------------

  static constexpr aer::F32 kInvChunkDim  = 1.0f / static_cast<aer::F32>(kChunkDim);

  // Chunk size in world-space
  static constexpr aer::F32 kChunkSize    = 10.0f; // 

  // Voxel size in world-space
  static constexpr aer::F32 kVoxelSize    = kChunkSize * kInvChunkDim;
  
  //---------------------

  // Size of the batch of chunks buffer
  static const aer::U32 kBufferBatchSize      = 350u;

  // There is a maximum of 5 triangles per voxels [do not change !]
  static const aer::U32 kMaxTrianglesPerVoxel = 5u;

  //-----------------------------------------------


  MarchingCubeRenderer() :
    nfreebuffers_(0u),
    grid_dim_(aer::Vector3(0u)),
    bInitialized_(false)
  {}

  ~MarchingCubeRenderer() {
    deinit();
  }

  void init();
  void deinit();

  void generate(const aer::Vector3 &grid_dimension);
  void render(const aer::Camera &camera);


 private:
  enum ChunkTriState_t {
    CHUNK_EMPTY,
    CHUNK_FILLED
  };

  struct ChunkInfo_t {
    aer::I32 id = -1;
    aer::I32 i, j, k;         // grid position (indices)
    aer::Vector3 ws_coords;    // world-space coordinates

    //aer::F32 distance_from_camera;
    ChunkTriState_t state;
  };

  void init_geometry();
  void init_textures();
  void init_buffers();
  void init_shaders();

  void create_chunk(ChunkInfo_t &chunk);
  void build_density_volume(ChunkInfo_t &chunk);
  void list_triangles();
  void generate_vertices(ChunkInfo_t &chunk);


  /// A sample counter query is used to know when passes generates no vertices.
  //aer::Query query_;

  struct {
    aer::Program build_density;
    aer::Program disp_density; // [for debugging]
    aer::Program trilist;
    aer::Program genvertices;
    aer::Program render_chunk;
  } program_;

  aer::Mesh slice_mesh_;

  struct {
    aer::Sampler nearest_clamp;
    aer::Sampler linear_clamp;
  } sampler_;

  aer::Texture3D   density_tex_;
  aer::Framebuffer density_rt_;

  aer::DeviceBuffer      trilist_base_buffer_;
  aer::DeviceBuffer      trilist_buffer_;
  aer::TransformFeedback trilist_tf_; //

  struct {
    aer::TBO_t case_to_numtri;
    aer::TBO_t edge_connect;
  } tbo_;

  //aer::Mesh grid_mesh_; //
  std::vector<aer::U32> freebuffer_indices_;
  aer::U32 nfreebuffers_;

  std::vector<aer::TransformFeedback> tf_stack_;

  std::vector<ChunkInfo_t> grid_;
  aer::Vector3 grid_dim_;


  aer::DeviceBuffer debug_vbos_[kBufferBatchSize]; // XXX tmp

  bool bInitialized_;
};

#endif  // MARCHING_CUBE_RENDERER_H_
