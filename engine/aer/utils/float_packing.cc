// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/utils/float_packing.h"


namespace aer {

U32 PackUnitFloat(F32 fvalue, U32 nBits) {
  AER_CHECK((fvalue >= 0.0f) && (fvalue <= 1.0f));

  U32 boundValue = 1u << nBits;
  F32 scaled = fvalue * static_cast<F32>(boundValue - 1u);

  U32 rounded = static_cast<U32>(scaled + 0.5f);
  rounded = (rounded > boundValue-1u) ? boundValue-1u : rounded;

  return rounded;
}

F32 UnpackUnitFloat(U32 uvalue, U32 nBits) {
  U32 boundValue = 1u << nBits;
  F32 rescale = 1.0f / static_cast<F32>(boundValue - 1u);

  F32 approx = rescale * static_cast<F32>(uvalue);
  return approx;
}

}  // namespace aer
