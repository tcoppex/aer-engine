// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_VIEW_FREE_CAMERA_H_
#define AER_VIEW_FREE_CAMERA_H_

#include "aer/common.h"
#include "aer/view/camera.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Describes a functional camera to fly through a scene,
/// with events handling for Mouse+Keyboard and PS3 SixAxis.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class FreeCamera : public Camera {
 public:
  FreeCamera(const View &view, const Frustum &frustum);

  void update();

  /// Getters
  F32  motion_factor()        const { return motion_factor_;    }
  F32  rotation_factor()      const { return rotation_factor_;  }
  F32  inertia_factor()       const { return inertia_factor_;   }
  bool is_x_axis_limited()    const { return bLimitPitchAngle_; }
  bool is_x_axis_inverted()   const { return bInvertYaw_;       }
  bool is_y_axis_inverted()   const { return bInvertPitch_;     }
  bool is_motion_enabled()    const { return bEnableMotion_;    }
  bool is_rotation_enabled()  const { return bEnableRotation_;  }
  bool is_joystick_used()     const { return bUseJoystick_;     }

  /// Setters
  void set_motion_factor(const F32 factor)   { motion_factor_   = factor; }
  void set_rotation_factor(const F32 factor) { rotation_factor_ = factor; }
  void set_inertia_factor(const F32 factor)  { inertia_factor_  = factor; }

  void limit_x_axis(bool state)        { bLimitPitchAngle_  = state; }
  void invert_x_axis(bool state)       { bInvertPitch_      = state; }
  void invert_y_axis(bool state)       { bInvertYaw_        = state; }
  void enable_motion(bool state)       { bEnableMotion_     = state; }
  void enable_motion_noise(bool state) { bMotionNoise_      = state; }
  void enable_rotation(bool state)     { bEnableRotation_   = state; }
  void use_joystick(bool state)        { bUseJoystick_      = state; }

  // Override view's setters to update Euler's angles
  void set_position(const Vector3& position) {
    Camera::set_position(position);
    update_euler_angles();
  }

  void set_direction(const Vector3& direction) {
    Camera::set_direction(direction);
    update_euler_angles();
  }

  void set_target(const Vector3& target) {
    Camera::set_target(target);
    update_euler_angles();
  }

  void set_view(const View& view) {
    Camera::set_view(view);
    update_euler_angles();
  }

  //void set_tracking_target(const Position3DInterface& target);

 private:
  void update_motion();
  void update_rotation();
  void update_euler_angles();

  F32 pitch_angle_;                 /// x-axis rotation angle (in radians)
  F32 yaw_angle_;                   /// y-axis rotation angle (in radians)

  F32 motion_factor_;
  F32 rotation_factor_;
  F32 inertia_factor_;
  Vector3 motion_velocity_;
  Vector2 rotation_velocity_;

  Vector2 cursor_delta_;

  bool bLimitPitchAngle_;           /// when true, limits pitch to [-pi/2, pi/2]
  bool bInvertPitch_;               /// when true, inverts the pitch
  bool bInvertYaw_;                 /// when true, inverts the yaw
  bool bEnableRotation_;
  bool bEnableMotion_;
  bool bUseJoystick_;

  bool bMotionNoise_;               /// experimental effect
};

}  // namespace aer

#endif  // AER_VIEW_FREE_CAMERA_H_
