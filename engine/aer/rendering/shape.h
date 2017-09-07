// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_RENDERING_SHAPE_H_
#define AER_RENDERING_SHAPE_H_

#include "aer/common.h"
#include "aer/rendering/drawable.h"
#include "aer/rendering/mesh.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// A shape represent a 2d or 3d drawable primitive
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class Shape : public Drawable {
 public:
  /// Specify default location used for attributes
  enum AttributeLocation_t {
    POSITION = 0,
    NORMAL,
    TEXCOORD
  };

  void draw() const override {
    mesh_.draw();
  }

  void draw_instances(const U32 count) const override {
    mesh_.draw_instances(count);
  }

 protected:
  Mesh mesh_;
};

class Plane : public Shape {
 public:
  void init(const F32 width, const F32 height, const U32 resolution);
};

class Cube : public Shape {
 public:
  void init(const F32 length);
};

class SphereRaw : public Shape {
 public:
  void init(const F32 radius, const U32 resolution);
};

class Dome : public Shape {
 public:
  void init(const F32 radius, const U32 resolution);
};

}  // namespace aer

#endif  // AER_RENDERING_SHAPE_H_
