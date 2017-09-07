file(GLOB_RECURSE Sources *.cc)
file(GLOB_RECURSE Headers *.h)

find_package(OpenGL)
find_package(SFML)
find_package(GLEW 0.9.0 REQUIRED)
find_package(OpenMP)

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS           "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB GLSW_Sources ${THIRDPARTIES}/glsw/*.c)
file(GLOB GLSW_Headers ${THIRDPARTIES}/glsw/*.h)
set(Sources ${Sources};${GLSW_Sources})
set(Headers ${Headers};${GLSW_Headers})
source_group(GLSW FILES ${GLSW_Sources} ${GLSW_Headers})

set(WARNING_FLAGS "-Wall -Wextra -Wpedantic -Wcast-align -Wundef")
set(EXTRA_FLAGS "-ftree-loop-distribution -funroll-loops")
set(CMAKE_CXX_FLAGS          "-std=gnu++11 ${WARNING_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE  "-O2 -DNDEBUG") #-msse (to enable SSE instruction)
set(CMAKE_CXX_FLAGS_DEBUG    "-O0 -g")

set(TARGET_NAME aer)

add_library(
  ${TARGET_NAME} STATIC
  ${Sources} 
  ${Headers}
)

target_link_libraries(${TARGET_NAME} 
  ${GLEW_LIBRARIES}
)
