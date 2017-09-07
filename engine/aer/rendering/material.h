// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_RENDERING_MATERIAL_H_
#define AER_RENDERING_MATERIAL_H_

#include <string>
#include "aer/common.h"


namespace aer {

class Texture2D;

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Handle the set of materials of a renderable object.
/// This class is intend to be redesign completely.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class Material {
 public:
  struct PhongAttributes_t {
    Vector3 diffuse;
    Vector3 specular;
    F32     shininess;
  };

  struct BlendAttributes_t {
    F32 factor;
  };


  inline Material();

  /// Getters
  inline const PhongAttributes_t& phong_attributes() const;
  inline const BlendAttributes_t& blend_attributes() const;

  inline Texture2D* diffuse_map() const;
  inline Texture2D* specular_map() const;
  inline Texture2D* normal_map() const;

  inline bool has_diffuse_map() const;
  inline bool has_specular_map() const;
  inline bool has_normal_map() const;

  inline void set_phong_diffuse(const Vector3 &v);
  inline void set_phong_specular(const Vector3 &v);
  inline void set_phong_shininess(F32 shininess);

  inline void set_blend_factor(F32 factor);

  inline void set_diffuse_map_id(const std::string &id);
  inline void set_specular_map_id(const std::string &id);
  inline void set_normal_map_id(const std::string &id);


 private:
  PhongAttributes_t phong_attribs_;
  BlendAttributes_t blend_attribs_;

  // Contains ID to retrieve maps
  struct {
    Texture2D* diffuse;
    Texture2D* specular;
    Texture2D* normal;
  } map_id_;
};

}  // namespace aer

#include "aer/rendering/material-inl.h"

#endif  // AER_RENDERING_MATERIAL_H_
