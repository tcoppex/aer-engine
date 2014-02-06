// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_VIEW_FRUSTUM_H_
#define AER_VIEW_FRUSTUM_H_

#include "aer/common.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
/// 
/// Represents projectives parameters
/// 
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class Frustum {
 public:
  enum PlaneSide
  {
    PLANE_LEFT,
    PLANE_RIGHT,
    PLANE_TOP,
    PLANE_BOTTOM,      
    PLANE_NEAR,
    PLANE_FAR,
    kNumPlane
  };

  static 
  inline void ExtractPlanes(const Matrix4x4& matrix, 
                            const bool bNormalize,
                            Vector4 planes[kNumPlane]);

  inline Frustum();
  inline Frustum(F32 fov, F32 aspectRatio, F32 zNear, F32 zFar);

  /// Getters
  F32 fov()          const { return fov_; }
  F32 aspect_ratio() const { return aspect_ratio_; }
  F32 znear()        const { return znear_; }
  F32 zfar()         const { return zfar_; }

  const Vector2& linearization_params() const { return linearization_params_; }

  inline const Matrix4x4& projection_matrix() const;
  inline const Matrix4x4& projection_matrix();
  inline const Matrix4x4& inverse_projection_matrix() const;
  inline const Matrix4x4& inverse_projection_matrix();

  bool is_dirty() const { return bRebuild_; }


  /// Setters
  inline void set_fov(const F32 fov);
  inline void set_aspect_ratio(const F32 aspect_ratio);
  inline void set_znear(const F32 znear);
  inline void set_zfar(const F32 zfar);


 private:
  inline void build();
  inline void build_inverse();
  inline void update_linearization_params();

  F32 fov_;
  F32 aspect_ratio_;
  F32 znear_;
  F32 zfar_;

  Vector2 linearization_params_;

  Matrix4x4 projection_matrix_;
  Matrix4x4 inverse_projection_matrix_;

  bool bRebuild_;
  bool bRebuildInverse_;
};

}  // namespace aer

#include "aer/view/frustum-inl.h"

#endif  // AER_VIEW_FRUSTUM_H_
