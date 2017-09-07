// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/app/window.h"
#include "aer/core/opengl.h"


namespace aer {

Window::Window(const Display_t& display, const char *title)
  : display_(display),
    bClosed_(false)
{
  sf::ContextSettings settings;

  settings.depthBits          = display.depth_bits;
  settings.stencilBits        = display.stencil_bits;
  settings.antialiasingLevel  = display.msaa_level;
  settings.majorVersion       = display.gl_major;
  settings.minorVersion       = display.gl_minor;

  sf::Uint32 styles = sf::Style::None;
  if (display.bFullscreen) styles |= sf::Style::Fullscreen;
  if (display.bResizable)  styles |= sf::Style::Resize;
  if (display.bBorder)     styles |= sf::Style::Titlebar;
  if (display.bClose)      styles |= sf::Style::Close;

  handle_.create(sf::VideoMode(display.width, display.height),
                 title, styles, settings);
  set_verticalsync(false);

  aer::opengl::Initialize();
}

Window::~Window() {
  close();
}

bool Window::pollevent(sf::Event &ev) {
  return handle_.pollEvent(ev);
}

void Window::close() {
  handle_.close();
  bClosed_ = true;
}

void Window::swap() {
  handle_.display();
}

void Window::set_title(const char* title) {
  handle_.setTitle(title);
}

void Window::set_position(const I32 x, const I32 y) {
  handle_.setPosition(sf::Vector2i(x, y));
}

void Window::resize(const U32 w, const U32 h) {
  if ((w != display_.width) || (h != display_.height)) {
    display_.width  = w;
    display_.height = h;
    handle_.setSize(sf::Vector2u(w, h));
  }
}

void Window::set_verticalsync(bool status) {
  handle_.setVerticalSyncEnabled(status);
}

void Window::set_fullscreen(bool status) {
  AER_ASSERT("not implemented yet" && false);
}

void Window::set_visibility(bool status) {
  handle_.setVisible(status);
}

}  // namespace aer
