// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_COMPUTE_PROGRAM_H_
#define AER_DEVICE_COMPUTE_PROGRAM_H_

#include "aer/common.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// share most of its attribute with traditional programs
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class ComputeProgram : public AbstractProgram {
 public:
  void launch(const Vector3i &blockDim);

  //void update_define(const std::string &name, F32 fvalue);
  //void update_define(const std::string &name, U32 uvalue);
  //bool compile();
};

}  // namespace aer

#endif  // AER_DEVICE_COMPUTE_PROGRAM_H_
