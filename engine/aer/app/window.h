// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_APP_WINDOW_H_
#define AER_APP_WINDOW_H_

/*
Et peut-être qu'un jour il comprendrait,
qu'après tout, 
il y avait là une raison inconnue.
*/

#include "SFML/Window.hpp"

#include "aer/common.h"
#include "aer/app/display.h"

namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
/// 
/// 
/// 
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class Window {
 public:
  Window(const Display_t& display, const char *title);
  ~Window();

  bool pollevent(sf::Event &ev);
  void close();
  void swap();

  /// Getters
  const Display_t& display() const { 
    return display_; 
  }

  bool has_closed() const {
    return bClosed_;
  }

  /// Setters
  void set_title(const char* title);
  void set_position(const I32 x, const I32 y);
  void resize(const U32 w, const U32 h);
  void set_verticalsync(bool status);
  void set_fullscreen(bool status);
  void set_visibility(bool status);


 private:
  sf::Window handle_;
  Display_t display_;
  bool bClosed_;

  DISALLOW_COPY_AND_ASSIGN(Window);
};

}  // namespace aer

#endif  // AER_APP_WINDOW_H_
