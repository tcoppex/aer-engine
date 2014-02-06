// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_DEBUG_RENDERER_H_
#define AER_DEVICE_DEBUG_RENDERER_H_

#include "aer/common.h"
#include "aer/utils/singleton.h"

namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class DebugRenderer : public Singleton<DebugRenderer> {
 public:
  void add_line(const Vector3 &point_src,
                const Vector3 &point_dst,
                F32 width = 1.0f,
                F32 duration = 0.0f,
                bool bUseDepth = true);

  void add_axes(const Matrix4x4& model,
                F32 size = 1.0f,
                F32 duration = 0.0f,
                bool bUseDepth = true);

  void add_circle(const Vector3 &center,
                  const Vector3 &direction,
                  F32 radius,
                  F32 duration = 0.0f,
                  bool bUseDepth = true);

  void add_sphere(const Vector3 &center,
                  F32 radius,
                  F32 duration = 0.0f,
                  bool bUseDepth = true);

  void add_AABB(const Vector3 &min_coords,
                const Vector3 &max_coords,
                F32 width = 1.0f,
                F32 duration = 0.0f,
                bool bUseDepth = true);

  //void add_OBB();


 private:
  // [TODO] stack of object to render

  friend Singleton<DebugRenderer>;
};

}  // namespace aer

#endif  // AER_DEVICE_DEBUG_RENDERER_H_
