// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_TRANSFORM_FEEDBACK_H_
#define AER_DEVICE_TRANSFORM_FEEDBACK_H_

#include "aer/common.h"
#include "aer/device/device_resource.h"

namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
///
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class TransformFeedback : public DeviceResource {
 public:
  static
  void Unbind() {
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0u);
  }

  void generate() {
    AER_ASSERT(!is_generated());
    glGenTransformFeedbacks(1, &id_);
  }

  void release() {
    if (is_generated()) {
      glDeleteTransformFeedbacks(1, &id_);
      id_ = 0u;
    }
  }

  void bind() const {
    AER_ASSERT(is_generated());
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, id_);
  }

  void unbind() const {
    Unbind();
  }

  //---------------------------------
  static
  void begin(GLenum primitive_mode) {
    glBeginTransformFeedback(primitive_mode);
  }

  static
  void end() {
    glEndTransformFeedback();
  }

  static
  void pause() {
    glPauseTransformFeedback();
  }

  static
  void resume() {
    glResumeTransformFeedback();
  }
  //---------------------------------

  void draw(GLenum mode) {
    glDrawTransformFeedback(mode, id_);
  }

  void draw_instanced(GLenum mode, aer::I32 count) {
    glDrawTransformFeedbackInstanced(mode, id_, count);
  }

  void draw_stream(GLenum mode, aer::U32 stream) {
    glDrawTransformFeedbackStream(mode, id_, stream);
  }

  void draw_stream_instanced(GLenum mode, aer::U32 stream, aer::I32 count) {
    glDrawTransformFeedbackStreamInstanced(mode, id_, stream, count);
  }
};
  
}  // namespace aer

#endif  // AER_DEVICE_TRANSFORM_FEEDBACK_H_
