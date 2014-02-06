aer-engine
================

About
---------------------------------

An OpenGL 4.3 / C++ 11 rendering engine oriented towards animation.


Demos
---------------------------------
<table>
    <tr>
        <td>aura</td>
        <td>A technical demo demonstrating the animation capabilities of the engine, with some rendering techniques.</td>
    </tr>
    <tr>
        <td>cuda_cs_blur</td>
        <td>Performance comparison between a CUDA and a Compute Shader blur kernel.</td>
    </tr>
    <tr>
        <td>gpu_raymarching</td>
        <td>Raymarching on a Fragment Shader.</td>
    </tr>
    <tr>
        <td>marching_cube</td>
        <td>  Procedural geometry generation with a marching cube algorithm on the GPU using
  tesselation feedback.</td>
    </tr>
    <tr>
        <td>simple_hair</td>
        <td>  Hair simulation rendered with Tesselation.</td>
    </tr>
</table>

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


