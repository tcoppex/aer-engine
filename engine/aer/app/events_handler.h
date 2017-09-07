// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_APP_EVENTS_HANDLER_H_
#define AER_APP_EVENTS_HANDLER_H_

#include <algorithm>
#include <set>
#include "SFML/Window.hpp" // for sf::Event

#include "aer/common.h"
#include "aer/utils/singleton.h"
#include "aer/app/event_button.h"  // for the buttons id


namespace aer {

class Window;

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
/// 
/// Store per-frame events to be handled by the app
/// 
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class EventsHandler : public Singleton<EventsHandler> {
 public:
  void update(Window &window);

  /// Keyboard events
  bool key_pressed  (const Keyboard::Key key) const;
  bool key_released (const Keyboard::Key key) const;
  bool key_down     (const Keyboard::Key key) const;
  bool key_up       (const Keyboard::Key key) const;

  /// Mouse events
  bool mouse_button_pressed (const Mouse::Button button) const;
  bool mouse_button_released(const Mouse::Button button) const;
  bool mouse_button_down    (const Mouse::Button button) const;
  bool mouse_button_up      (const Mouse::Button button) const;

  /// Joystick events
  bool joystick_button_pressed (const U32 button) const;
  bool joystick_button_released(const U32 button) const;
  bool joystick_button_down    (const U32 button) const;
  bool joystick_button_up      (const U32 button) const;
  
  const Vector2i& mouse_position() const {
    return mouse_.position;
  }
  
  const Vector2& mouse_delta() const {
    return mouse_.delta;
  }

  /// Joystick events
  F32 joystick_axis_position(I32 axis) const {
    return joystick_.position[axis]; 
  }

  bool is_joystick_connected() const { 
    return bJoystickConnected_; 
  }

  /// Window events
  bool has_closed()       const { return bClosed_; }
  bool has_resized()      const { return bResized_; }
  bool has_gained_focus() const { return bGainedFocus_; }
  bool has_lost_focus()   const { return bLostFocus_; }

  const Vector2i& window_size() const {
    return windowevent_.size;
  }


 private:
  typedef std::set<U32>       ButtonSet;
  typedef ButtonSet::iterator ButtonSetIterator;

  struct MouseMotion {
    Vector2i position;
    Vector2i last_position;
    Vector2  delta;
  };

  struct JoystickMotion {
    Vector4 position;
  };

  struct WindowEvent {
    Vector2i size;
  };


  EventsHandler();

  void reset();

  void dispatch_event();

  void handle_close();
  void handle_resize();
  void handle_focus(bool bFocus);
  void handle_keyboard(bool bPressed);
  void handle_mouse_button(bool bPressed);
  void handle_mouse_motion();
  //void handle_mouse_wheel();
  //void handle_mouse_focus(bool bGainFocus);
  void handle_joystick_button(bool bPressed);
  void handle_joystick_motion();
  void handle_joystick_connection(bool state);

  
  sf::Event       event_;

  ButtonSet       key_pressed_;
  ButtonSet       key_down_;
  ButtonSet       key_released_;

  ButtonSet       mouse_button_pressed_;
  ButtonSet       mouse_button_down_;
  ButtonSet       mouse_button_released_;

  ButtonSet       joystick_button_pressed_;
  ButtonSet       joystick_button_down_;
  ButtonSet       joystick_button_released_;

  MouseMotion     mouse_;
  JoystickMotion  joystick_;
  WindowEvent     windowevent_;

  bool            bClosed_;
  bool            bResized_;
  bool            bGainedFocus_;
  bool            bLostFocus_;
  bool            bJoystickConnected_;

  /// Friends
  friend class Singleton<EventsHandler>;
};

//template<class T> T* Singleton<T>::sInstance = nullptr;

}  // namespace aer

#endif  // AER_APP_EVENTS_HANDLER_H_
