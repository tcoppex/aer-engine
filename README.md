aer-engine
================

About
---------------------------------

An OpenGL 4.3 / C++ 11 rendering engine oriented towards animation.


Demos
---------------------------------

* aura :
  A technical demo demonstrating the animation capabilities of the engine, with
  some rendering techniques.
  
* cuda_cs_blur :
  Performance comparison between a CUDA and a Compute Shader blur kernel.

* gpu_raymarching :
  Raymarching on a Fragment Shader.
  
* marching_cube :
  Procedural geometry generation with a marching cube algorithm on the GPU using
  tesselation feedback.

* simple_hair :
  Hair simulation rendered with Tesselation.


Compilation
---------------------------------

Compile first the engine, then the demos :
```
mkdir build
cd build
mkdir engine demos
cd engine
cmake ../../engine -DCMAKE_BUILD_TYPE:STRING=Release
make -j4
cd ../demos
cmake ../../demos -DCMAKE_BUILD_TYPE:STRING=Release
make -j4
```


