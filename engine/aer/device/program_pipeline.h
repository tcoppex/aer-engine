// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_PROGRAM_PIPELINE_H_
#define AER_DEVICE_PROGRAM_PIPELINE_H_

#include "aer/common.h"
#include "aer/device/device_resource.h"
#include "aer/device/program.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
///
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class ProgramPipeline : public DeviceResource {
 public:
  ProgramPipeline() :
    DeviceResource(),
    bitfield_(0u)
  {}


  void generate() {
    AER_ASSERT(!is_generated());
    glGenProgramPipelines(1u, &id_);
  }

  void release() {
    if (is_generated()) {
      glDeleteProgramPipelines(1u, &id_);
      id_ = 0u;
    }
  }

  void bind() const {
    Program::Deactivate(); //
    glBindProgramPipeline(id_);
  }

  void unbind() const {
    glBindProgramPipeline(0u);
  }

  /// Use a program stages as part of the current pipeline
  /// @param stages : bitfield specifying the program stages to bind
  /// @param program : pointer to a program to bind, or nullptr to unbind
  void use_program_stages(GLbitfield stages, const Program *program) {
    if (!program) {
      bitfield_ &= ~stages;
      glUseProgramStages(id_, stages, 0u);
      return;
    }

    AER_ASSERT(program->is_separable());

    bitfield_ |= stages;
    glUseProgramStages(id_, stages, program->id());
  }

  /// Return the current stages in use
  GLbitfield bitfield() const {
    return bitfield_;
  }


 private:
  GLbitfield bitfield_;
};

}  // namespace aer

#endif  // AER_DEVICE_PROGRAM_PIPELINE_H_
