// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_AER_H_
#define AER_AER_H_

/*
Il ne devait guère s'être écoulé plus d'un an depuis que, dans le jardin du 
château qui s'étendait vers la mer en une pente assez abrupte, quelque chose 
d'étonnant lui était arrivé. Allant et venant avec un livre, selon son habitude, 
il en était venu à prendre appui, à peu près à hauteur d'épaule, dans la fourche 
d'un arbre ramifié, et aussitôt, il sentit que cette position lui procurait un 
soutien si agréable, une telle abondance de repos, qu'il resta ainsi, sans lire,
complétement enchâssé dans la nature, en une contemplation presque inconsciente.
*/

/// includes all core headers
#include "aer/common.h"
#include "aer/core/opengl.h"

// Singletons
#include "aer/app/events_handler.h"
#include "aer/utils/global_clock.h"
#include "aer/utils/logger.h"
#include "aer/loader/shader_proxy.h"
#include "aer/loader/texture_2d_proxy.h"


namespace aer {
class Application;
class Display;
class Window;
class EventsHandler;

class VertexArray;
class Sampler;
class Texture;
class Texture2D;
class TextureBuffer;
class Shader;
class Program;
class ProgramPipeline;
class DeviceBuffer;
class Framebuffer;

class ShaderProxy;
class Texture2DProxy;

class PoolAllocator;
class StackAllocator;

class ParticleSystem;
class VerletIntegrator;

class Material;
class Mesh;

class GlobalClock;
class Timer;

class Camera;
class FreeCamera;
class Frustum;
class View;
} // namespace aer


#endif  // AER_AER_H_
