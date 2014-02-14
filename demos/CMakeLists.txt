cmake_minimum_required(VERSION 2.8)


# ------------
# - COMPILER -
# ------------
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

#cmake .. -DCMAKE_BUILD_TYPE:STRING=Release
set(CMAKE_CXX_FLAGS          "-std=gnu++11")
set(CMAKE_CXX_FLAGS_RELEASE  "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG    "-O0 -g -Werror -pedantic")


# -------------
# - LIBRARIES -
# -------------
set(AER_PATH ${CMAKE_SOURCE_DIR}/../engine)
set(AER_INCLUDE_DIRS ${AER_PATH}/aer)
set(AER_LIBRARY_DIRS ${AER_PATH}/lib)
include_directories(${AER_PATH})
include_directories(${CMAKE_SOURCE_DIR})

set(AER_LIBRARY  ${AER_LIBRARY_DIRS}/libaer.a)
set(GLEW_LIBRARY ${AER_LIBRARY_DIRS}/libGLEW.a)
find_package(OpenGL)
find_package(SFML)

set(LINKED_LIBRARIES ${AER_LIBRARY}
                     ${GLEW_LIBRARY}
                     ${OPENGL_gl_LIBRARY}
                     sfml-window
                     sfml-system
                     freeimage)

# --------------
# - THIRDPARTY -
# --------------
set(THIRDPARTY_INCLUDE_DIRS ${AER_PATH}/thirdparty)
include_directories(${THIRDPARTY_INCLUDE_DIRS})

add_subdirectory(aura)
add_subdirectory(cuda_cs_blur)
add_subdirectory(gpu_raymarching)
add_subdirectory(marching_cube)
add_subdirectory(simple_hair)
