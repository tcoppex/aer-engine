// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_UTILS_FLOAT_PACKING_H_
#define AER_UTILS_FLOAT_PACKING_H_

#include "aer/common.h"


namespace aer {

/// fvalue must be in the range [0, 1]
U32 PackUnitFloat(F32 fvalue, U32 nBits);

F32 UnpackUnitFloat(U32 uvalue, U32 nBits);

// TODO
//U32 PackFloat(F32 value, F32 min, F32 max, U32 nBits);
//F32 UnpackFloat(U32 value, F32 min, F32 max, U32 nBits);

}  // namespace aer

#endif  // AER_UTILS_FLOAT_PACKING_H_
