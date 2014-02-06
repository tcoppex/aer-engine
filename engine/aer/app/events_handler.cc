// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/app/events_handler.h"
#include "aer/app/window.h"


namespace aer {

EventsHandler::EventsHandler() :
  bClosed_(false),
  bResized_(false),
  bGainedFocus_(false),
  bLostFocus_(false),
  bJoystickConnected_(false)
{
  key_pressed_.clear();
  key_down_.clear();
  key_released_.clear();
  mouse_button_pressed_.clear();
  mouse_button_down_.clear();
  mouse_button_released_.clear();
  joystick_button_pressed_.clear();
  joystick_button_down_.clear();
  joystick_button_released_.clear();
}

void EventsHandler::update(Window &window) {
  reset();
  while (window.pollevent(event_)) {
    dispatch_event();
  }
}

bool EventsHandler::key_pressed(const Keyboard::Key key) const {
  return key_pressed_.find(key) != key_pressed_.end();
}

bool EventsHandler::key_released(const Keyboard::Key key) const {
  return (key_released_.find(key) != key_released_.end());
}

bool EventsHandler::key_down(const Keyboard::Key key) const {
  return key_down_.find(key) != key_down_.end();
}

bool EventsHandler::key_up(const Keyboard::Key key) const {
  return !key_down(key);
}


bool EventsHandler::mouse_button_pressed(const Mouse::Button button) const {
  return mouse_button_pressed_.find(button) != mouse_button_pressed_.end();
}

bool EventsHandler::mouse_button_released(const Mouse::Button button) const {
  return mouse_button_released_.find(button) != mouse_button_released_.end();
}

bool EventsHandler::mouse_button_down(const Mouse::Button button) const {
  return mouse_button_down_.find(button) != mouse_button_down_.end();
}

bool EventsHandler::mouse_button_up(const Mouse::Button button) const {
  return !mouse_button_down(button);
}


bool EventsHandler::joystick_button_pressed(const U32 button) const {
  return joystick_button_pressed_.find(button) != joystick_button_pressed_.end();
}

bool EventsHandler::joystick_button_released(const U32 button) const {
  return joystick_button_released_.find(button) != joystick_button_released_.end();
}

bool EventsHandler::joystick_button_down(const U32 button) const {
  return joystick_button_down_.find(button) != joystick_button_down_.end();
}

bool EventsHandler::joystick_button_up(const U32 button) const {
  return !joystick_button_down(button);
}


void EventsHandler::reset() {
  // ----- costly ?
  key_pressed_.clear();
  mouse_button_pressed_.clear();
  joystick_button_pressed_.clear();

  key_released_.clear();
  mouse_button_released_.clear();
  joystick_button_released_.clear();
  // -----

  mouse_.delta = Vector2(0.0f);
  
  bClosed_      = false;
  bResized_     = false;
  bGainedFocus_ = false;
  bLostFocus_   = false;
}

void EventsHandler::dispatch_event() {
  switch (event_.type)
  {
    case sf::Event::Closed:
      handle_close();
    break;
    
    case sf::Event::Resized:
      handle_resize();
    break;
    
    case sf::Event::LostFocus:
      handle_focus(false);
    break;
    
    case sf::Event::GainedFocus:      
      handle_focus(true);
    break;
    
    case sf::Event::TextEntered:
      // Empty
    break;
    
    case sf::Event::KeyPressed:
      handle_keyboard(true);
    break;
    
    case sf::Event::KeyReleased:      
      handle_keyboard(false);
    break;
    
    case sf::Event::MouseButtonPressed:
      handle_mouse_button(true);
    break;
    
    case sf::Event::MouseButtonReleased:
      handle_mouse_button(false);
    break;
    
    case sf::Event::MouseMoved:
      handle_mouse_motion();
    break;
    
    case sf::Event::MouseWheelMoved:
      //handle_wheel_motion();
    break;
    
    case sf::Event::MouseEntered:
      //handle_mouse_focus(true);
    break;
    
    case sf::Event::MouseLeft:
      //handle_mouse_focus(false);
    break;
    
    case sf::Event::JoystickButtonPressed:
      handle_joystick_button(true);
    break;
    
    case sf::Event::JoystickButtonReleased:
      handle_joystick_button(false);
    break;
    
    case sf::Event::JoystickMoved:
      handle_joystick_motion();
    break;
    
    case sf::Event::JoystickConnected:
      handle_joystick_connection(true);
    break;
    
    case sf::Event::JoystickDisconnected:
      handle_joystick_connection(false);
    break;

    default:
      AER_CHECK("** unknown event triggered **");
    break;
  }
}


void EventsHandler::handle_close() {
  bClosed_ = true;
}

void EventsHandler::handle_resize() {
  bResized_ = true;
  windowevent_.size = Vector2i(event_.size.width, event_.size.height);
}

void EventsHandler::handle_focus(bool bFocus) {
  if (bFocus) {
    bGainedFocus_ = true;
  } else {
    bLostFocus_ = true;
  }
}

void EventsHandler::handle_keyboard(bool bPressed) {
  const Keyboard::Key key = Keyboard::Key(event_.key.code);

  if (bPressed) {
    if (!key_down(key)) {
      key_pressed_.insert(key);
    }
    key_down_.insert(key);
  } else {
    key_released_.insert(key);

    // remove the key from KeyPressed
    ButtonSetIterator it = key_down_.find(key);
    if (it != key_down_.end()) {
      key_down_.erase(it);
    }
  }
}

void EventsHandler::handle_mouse_button(bool bPressed) {
  const Mouse::Button button = Mouse::Button(event_.mouseButton.button);

  if (bPressed) {
    if (!mouse_button_down(button)) {
      mouse_button_pressed_.insert(button);
    }
    mouse_button_down_.insert(button);
  } else {
    mouse_button_released_.insert(button);

    // remove the key from MouseButtonPressed
    ButtonSetIterator it = mouse_button_down_.find(button);
    if (it != mouse_button_down_.end()) {
      mouse_button_down_.erase(it);
    }
  }
}

void EventsHandler::handle_mouse_motion() {
  mouse_.last_position  = mouse_.position;
  mouse_.position       = Vector2i(event_.mouseMove.x, event_.mouseMove.y);
  mouse_.delta          = mouse_.position - mouse_.last_position;
}

void EventsHandler::handle_joystick_button(bool bPressed) {
  U32 joystick_id = event_.joystickButton.joystickId;
  U32 button      = event_.joystickButton.button;
  
  //printf("joystick %d pressed button %d\n", joystick_id, button);
  AER_CHECK(joystick_id == 0u);

  if (bPressed) {
    if (!joystick_button_down(button)) {
      joystick_button_pressed_.insert(button);
    }
    joystick_button_down_.insert(button);
  } else {
    joystick_button_released_.insert(button);

    ButtonSetIterator it = joystick_button_down_.find(button);
    if (it != joystick_button_down_.end()) {
      joystick_button_down_.erase(it);
    }
  }
}

void EventsHandler::handle_joystick_motion() {
  // Note : Handle only one joystick id

  if (event_.joystickMove.joystickId > 0) {
    AER_WARNING("joystick id not handled")
    return;
  }

  int axis = event_.joystickMove.axis;  
  if (axis >= 4u) {
    return;
  }

  // Scale the position from [-100, 100] to [-1.0f, 1.0f]
  joystick_.position[axis] = 0.01f * event_.joystickMove.position;
}

void EventsHandler::handle_joystick_connection(bool state) {
  bJoystickConnected_ = state;
}



}  // namespace aer
