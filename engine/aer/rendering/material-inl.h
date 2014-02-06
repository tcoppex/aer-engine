// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_RENDERING_MATERIAL_INL_H_
#define AER_RENDERING_MATERIAL_INL_H_

#include "aer/loader/texture_2d_proxy.h"

namespace aer {

Material::Material() :
  map_id_{nullptr}
{
  phong_attribs_.shininess = 0.0f;
  blend_attribs_.factor    = 1.0f;
}

const Material::PhongAttributes_t& Material::phong_attributes() const {
  return phong_attribs_;
}

const Material::BlendAttributes_t& Material::blend_attributes() const {
  return blend_attribs_;
}

Texture2D* Material::diffuse_map() const {
  return map_id_.diffuse;
}

Texture2D* Material::specular_map() const {
  return map_id_.specular;
}

Texture2D* Material::normal_map() const {
  return map_id_.normal;
}

bool Material::has_diffuse_map() const {
  return map_id_.diffuse != nullptr; 
}

bool Material::has_specular_map()const {
  return map_id_.specular != nullptr; 
}

bool Material::has_normal_map() const {
  return map_id_.normal != nullptr; 
}

void Material::set_phong_diffuse(const Vector3 &v) {
  phong_attribs_.diffuse = v;
}

void Material::set_phong_specular(const Vector3 &v) {
  phong_attribs_.specular = v;
}

void Material::set_phong_shininess(F32 shininess) {
  phong_attribs_.shininess = shininess;
}

void Material::set_blend_factor(F32 factor) {
  blend_attribs_.factor = factor;
}

void Material::set_diffuse_map_id(const std::string &id) {
  map_id_.diffuse = Texture2DProxy::Get().get(id);
}

void Material::set_specular_map_id(const std::string &id) {
  map_id_.specular = Texture2DProxy::Get().get(id);
}

void Material::set_normal_map_id(const std::string &id) {
  map_id_.normal = Texture2DProxy::Get().get(id);
}

}  // namespace aer

#endif  // AER_RENDERING_MATERIAL_INL_H_
