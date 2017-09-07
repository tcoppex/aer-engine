cmake_minimum_required(VERSION 2.8)

project(aer-demos)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake)

# ------------
# - COMPILER -
# ------------
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

#cmake .. -DCMAKE_BUILD_TYPE:STRING=Release

set(WARNING_FLAGS "-Wall -Wpedantic") #-Werror
set(CMAKE_CXX_FLAGS          "-std=gnu++11 ${WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE  "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG    "-O0 -g")

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# -------------
# - LIBRARIES -
# -------------
set(AER_PATH ${CMAKE_SOURCE_DIR}/../engine)
set(AER_INCLUDE_DIRS ${AER_PATH}/aer)
set(AER_LIBRARY_DIRS ${AER_PATH}/lib)
include_directories(${AER_PATH})
include_directories(${CMAKE_SOURCE_DIR})

set(AER_LIBRARY  ${AER_LIBRARY_DIRS}/libaer.a)

find_package(OpenGL REQUIRED)
find_package(SFML 2.1 EXACT REQUIRED)
find_package(GLEW 0.9 REQUIRED)

set(LINKED_LIBRARIES ${LINKED_LIBRARIES}
                     ${AER_LIBRARY}
                     ${GLEW_LIBRARIES}
                     ${OPENGL_gl_LIBRARY}
                     sfml-window
                     sfml-system
                     freeimage)

# --------------
# - THIRDPARTY -
# --------------
set(THIRDPARTY_INCLUDE_DIRS ${AER_PATH}/thirdparty)
include_directories(
  ${THIRDPARTY_INCLUDE_DIRS}
  ${GLEW_INCLUDE_DIRS}
)

add_subdirectory(aura)
add_subdirectory(cuda_cs_blur)
add_subdirectory(gpu_raymarching)
add_subdirectory(hair)
add_subdirectory(ik_demo)
add_subdirectory(marcheur)
add_subdirectory(marching_cube)
add_subdirectory(variance_shadow_mapping)
