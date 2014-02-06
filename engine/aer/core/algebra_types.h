// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_CORE_ALGEBRA_TYPES_H_
#define AER_CORE_ALGEBRA_TYPES_H_

#define GLM_SWIZZLE
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"


namespace aer {

typedef glm::vec2     Vector2;
typedef glm::ivec2    Vector2i;
typedef glm::vec3     Vector3;
typedef glm::ivec3    Vector3i;
typedef glm::vec4     Vector4;
typedef glm::ivec4    Vector4i;
typedef glm::mat3     Matrix3x3;
typedef glm::mat4     Matrix4x4;
typedef glm::mat4x3   Matrix4x3;

typedef glm::quat     Quaternion;  //

}  // namespace aer


#endif  // AER_CORE_ALGEBRA_TYPES_H_
