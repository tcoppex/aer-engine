// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_APP_DISPLAY_H_
#define AER_APP_DISPLAY_H_

#include "aer/common.h"

namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
/// 
/// Describes generic display attributes
/// 
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
struct Display_t {
  Display_t(const U32 width, const U32 height)
    : width(width),
      height(height),
      depth_bits(24u),
      stencil_bits(0u),
      msaa_level(0u),
      gl_major(4),
      gl_minor(3),
      bFullscreen(false),
      bResizable(false),
      bBorder(true),
      bClose(true)
  {}

  F32 aspect_ratio() const { 
    return static_cast<float>(width) / static_cast<float>(height); 
  }

  U32 width;
  U32 height;
  U32 depth_bits;
  U32 stencil_bits;
  U32 msaa_level;
  U32 gl_major;
  U32 gl_minor;  
  bool bFullscreen;
  bool bResizable;
  bool bBorder;
  bool bClose;
};

}  // namespace aer

#endif  // AER_APP_DISPLAY_H_
