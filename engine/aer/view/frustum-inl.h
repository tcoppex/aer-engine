// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_VIEW_FRUSTUM_INL_H_
#define AER_VIEW_FRUSTUM_INL_H_

#include <glm/gtc/matrix_transform.hpp> // for perspective
#include <glm/gtc/matrix_access.hpp>    // to get matrix row


namespace aer {

void Frustum::ExtractPlanes(const Matrix4x4& matrix,
                            const bool bNormalize,
                            Vector4 planes[kNumPlane]) {
  /// if matrix is the projection matrix, clipping plane are in view space
  /// if matrix is the MVP matrix, clipping plane are in model space

  const Matrix4x4& M = glm::transpose(matrix);

  planes[PLANE_LEFT]   = M[3] + M[0]; // Left
  planes[PLANE_RIGHT]  = M[3] - M[0]; // Right
  planes[PLANE_BOTTOM] = M[3] + M[1]; // Bottom
  planes[PLANE_TOP]    = M[3] - M[1]; // Top      
  planes[PLANE_NEAR]   = M[3] + M[2]; // Near
  planes[PLANE_FAR]    = M[3] - M[2]; // Far

  if (bNormalize) {
    for (U32 i=0u; i<kNumPlane; ++i) {
      planes[i] = glm::normalize(planes[i]);
    }
  }
}

Frustum::Frustum()
  : fov_(60.0f),
    aspect_ratio_(1.0f),
    znear_(0.1f),
    zfar_(1000.0f),
    bRebuild_(true),
    bRebuildInverse_(true)
{}

Frustum::Frustum(F32 fov, F32 aspectRatio, F32 zNear, F32 zFar) 
  : fov_(fov),
    aspect_ratio_(aspectRatio),
    znear_(zNear),
    zfar_(zFar),
    bRebuild_(true),
    bRebuildInverse_(true)
{}

const Matrix4x4& Frustum::projection_matrix() const {
  return const_cast<Frustum*>(this)->projection_matrix();
}

const Matrix4x4& Frustum::projection_matrix() {
  build();
  return projection_matrix_;
}

const Matrix4x4& Frustum::inverse_projection_matrix() const {
  return const_cast<Frustum*>(this)->inverse_projection_matrix();
}

const Matrix4x4& Frustum::inverse_projection_matrix() {
  build_inverse();
  return inverse_projection_matrix_;
}


void Frustum::set_fov(const F32 fov) {
  fov_ = fov;
  bRebuild_ = bRebuildInverse_ = true;
}

void Frustum::set_aspect_ratio(const F32 aspect_ratio) {
  aspect_ratio_ = aspect_ratio;
  bRebuild_ = bRebuildInverse_ = true;
}

void Frustum::set_znear(const F32 znear) {
  znear_ = znear;
  bRebuild_ = bRebuildInverse_ = true;
}

void Frustum::set_zfar(const F32 zfar) {
  zfar_ = zfar;
  bRebuild_ = bRebuildInverse_ = true;
}

void Frustum::build() {
  if (!is_dirty()) {
    return;
  }

  projection_matrix_ = glm::perspective(fov_, aspect_ratio_, znear_, zfar_);
  update_linearization_params();
  bRebuild_ = false;
  bRebuildInverse_ = true;
}

void Frustum::build_inverse() {
  if (!bRebuildInverse_) {
    return;
  }

  build();
  inverse_projection_matrix_ = glm::inverse(projection_matrix_);
  bRebuildInverse_ = false;
}

void Frustum::update_linearization_params() {
  linearization_params_.x = zfar_ / (zfar_ + znear_);
  linearization_params_.y = -znear_ * linearization_params_.x;
}

}  // namespace aer

#endif  // AER_VIEW_FRUSTUM_INL_H_
