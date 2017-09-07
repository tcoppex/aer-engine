// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------


#include "aer/view/free_camera.h"

#include "glm/gtc/noise.hpp"        // for simplex
#include "glm/gtx/euler_angles.hpp" // for yawPitchRoll
#include "aer/app/events_handler.h" // for EventsHandler
#include "aer/utils/global_clock.h" // for GlobalClock


namespace aer {

FreeCamera::FreeCamera(const View &view, const Frustum &frustum)
  : Camera(view, frustum),
    pitch_angle_(0.0f),
    yaw_angle_(0.0f),
    motion_factor_(0.0f),
    rotation_factor_(0.0f),
    inertia_factor_(0.0f),
    motion_velocity_(Vector3(0.0f)),
    rotation_velocity_(Vector2(0.0f)),
    cursor_delta_(Vector2(0.0f)),
    bLimitPitchAngle_(false),
    bInvertPitch_(false),
    bInvertYaw_(false),
    bEnableRotation_(false),
    bEnableMotion_(false),
    bUseJoystick_(false),
    bMotionNoise_(false)
{
  update_euler_angles();

  motion_factor_    = 0.50f;
  rotation_factor_  = 0.15f;
  inertia_factor_   = 0.85f;

  cursor_delta_     = Vector2(0.0f);

  bLimitPitchAngle_ = true;
  bInvertPitch_     = false;
  bInvertYaw_       = false;
  bEnableRotation_  = true;
  bEnableMotion_    = true;
  bUseJoystick_     = true;

  bMotionNoise_     = false;
}

void FreeCamera::update() {
  update_motion();
  update_rotation();

  F32 yaw   = yaw_angle_;
  F32 pitch = pitch_angle_;
  F32 roll  = 0.0f;

  // - experiment -
  if (bMotionNoise_) {
    /// noise to add to yaw & pitch to bring a natural feeling to the camera
    F32 t = 0.05f*GlobalClock::Get().relative_time(aer::SECOND);
    F32 n1 = 0.025f*glm::simplex(glm::vec2(sin(3.0f*t), cos(5.0f*t)));
    F32 n2 = 0.030f*glm::simplex(glm::vec2(sin(7.0f*t), cos(2.0f*t)));
    n1 *= glm::smoothstep(0.0f, 1.0f, 1.0f-abs(n1));
    n2 *= glm::smoothstep(0.0f, 1.0f, 1.0f-abs(n2));

    yaw   += n1;
    pitch += n2;
    roll  = 2.0f * n1 * n2;
  }

  // Compute the rotation matrix, ie. Ry(yaw) * Rx(pitch) * Rz(roll)
  glm::mat3 cam_rotation = glm::mat3(glm::yawPitchRoll(yaw, pitch, roll));

  const Vector3 front(0.0f, 0.0f, -1.0f);
  Vector3 direction = glm::normalize(cam_rotation * front);

  const Vector3 left(-1.0f, 0.0f, 0.0f);
  Vector3 world_left = glm::normalize(cam_rotation * left);

  // Compute the new view parameters
  Vector3 pos     = position() + cam_rotation * motion_velocity_;
  Vector3 target  = pos + direction;
  Vector3 up      = glm::cross(direction, world_left);

  view_.set(pos, target, up);
  build();  

  // NOTE:
  // Having the camera model matrix can be helpful 
  // + it holds position, target & up in its columns
  //camera = view⁻¹  
}

void FreeCamera::update_motion() {
  const EventsHandler &event = EventsHandler::Get();
  Vector3 v_direction = Vector3(0.0f);

  // == Joystick ==
  if (bUseJoystick_) {
    v_direction.x = event.joystick_axis_position(0);
    v_direction.z = event.joystick_axis_position(1);

    if (event.joystick_button_down(Joystick::R1))    v_direction.y += 1.0f;
    if (event.joystick_button_down(Joystick::R2))    v_direction.y -= 1.0f;
  }

  // == Keyboard ==
  if (event.key_down(Keyboard::D))        v_direction.x += 1.0f;
  if (event.key_down(Keyboard::Q))        v_direction.x -= 1.0f;
  if (event.key_down(Keyboard::PageUp))   v_direction.y += 1.0f;
  if (event.key_down(Keyboard::PageDown)) v_direction.y -= 1.0f;
  if (event.key_down(Keyboard::S))        v_direction.z += 1.0f;
  if (event.key_down(Keyboard::Z))        v_direction.z -= 1.0f;

  if ((v_direction.x != 0.0f) || 
      (v_direction.y != 0.0f) || 
      (v_direction.z != 0.0f)) {
    v_direction = glm::normalize(v_direction);
  }

  motion_velocity_ = motion_factor_ * v_direction;
}

namespace {

Vector2 lowpass_filter(Vector2 last_filtered_input, Vector2 raw_input, F32 rc) {
  // Note: rc time constant is equivalent to  1 / (2*PI*fc)
  // where fc is the cuttoff frequency
  F32 dt = static_cast<F32>(GlobalClock::Get().delta_time());
  F32 factor = dt / (rc + dt);
  return glm::mix(last_filtered_input, raw_input, factor);
}

}  // namespace

void FreeCamera::update_rotation() {
  const EventsHandler &event = EventsHandler::Get();

  if (!bEnableRotation_) {
    return;
  }

  bool bHasRotated = false;
  Vector2 new_delta(0.0f);

  /// Joystick
  if (bUseJoystick_) {
    const F32 joystick_factor = 7.0f; //
    new_delta.x = joystick_factor * event.joystick_axis_position(2);
    new_delta.y = joystick_factor * event.joystick_axis_position(3);

    if (glm::dot(new_delta, new_delta) >= FLT_EPSILON) {
      bHasRotated = true;
    }
  }

  /// Mouse
  if (event.mouse_button_released(Mouse::Left)) {
    cursor_delta_ = Vector2(0.0f);
  }
  if (event.mouse_button_down(Mouse::Left)) { 
    bHasRotated = true;
    new_delta = event.mouse_delta();
  }


  const F32 inertia = glm::dot(rotation_velocity_, rotation_velocity_);

  /// Update camera rotation
  if (bHasRotated || (inertia >= FLT_EPSILON)) {
    if (bHasRotated) {
      // interpolate to avoid jaggies
      cursor_delta_ = glm::mix(cursor_delta_, new_delta, 0.1f);
      rotation_velocity_ = glm::pow(rotation_factor_, 3.0f) * cursor_delta_;
    } else {
      // if the camera stop to move, add inertia.
      rotation_velocity_ *= inertia_factor_;
    }

    F32 yaw_delta = rotation_velocity_.x;
    yaw_angle_   += (bInvertYaw_)? yaw_delta : -yaw_delta;

    F32 pitch_delta = rotation_velocity_.y;
    pitch_angle_   += (bInvertPitch_)? pitch_delta : -pitch_delta;

    if (bLimitPitchAngle_) {
      F32 pi_2 = static_cast<F32>(M_PI_2);
      pitch_angle_ = glm::clamp(pitch_angle_, -pi_2, +pi_2);
    }
  }
}

void FreeCamera::update_euler_angles() {
  /// Retrieve the yaw & pitch angle
  glm::vec3 zAxis = - direction(); // also the third row of the viewMatrix

  yaw_angle_ = atan2f(zAxis.x, zAxis.z);

  F32 len = sqrtf(zAxis.x*zAxis.x + zAxis.z*zAxis.z);
  pitch_angle_ = - atan2f(zAxis.y, len);
}

} // aer
