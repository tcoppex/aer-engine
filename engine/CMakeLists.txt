cmake_minimum_required(VERSION 2.8)


project(aer)

set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

set(CMAKE_CXX_FLAGS          "-std=gnu++11")
set(CMAKE_CXX_FLAGS_RELEASE  "-O2 -DNDEBUG") #-msse (to enable SSE instruction)
set(CMAKE_CXX_FLAGS_DEBUG    "-O0 -g -Werror -pedantic")


set(THIRDPARTIES ${CMAKE_SOURCE_DIR}/thirdparty)
include_directories(${THIRDPARTIES})

set(AER_SOURCE_DIR aer)
include_directories(${AER_SOURCE_DIR} ".")

# -- static GLEW --
file(GLOB GLEW_Sources ${THIRDPARTIES}/GL/*.c)
file(GLOB GLEW_Headers ${THIRDPARTIES}/GL/*.h)
add_library(GLEW ${GLEW_Sources} ${GLEW_Headers})

add_subdirectory(aer)
