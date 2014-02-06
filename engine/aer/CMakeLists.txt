# Retrieve all source files
file(GLOB_RECURSE Sources *.cc)
file(GLOB_RECURSE Headers *.h)

find_package(OpenGL)
find_package(SFML)

# -- GLSW --
file(GLOB GLSW_Sources ${THIRDPARTIES}/glsw/*.c)
file(GLOB GLSW_Headers ${THIRDPARTIES}/glsw/*.h)
set(Sources ${Sources};${GLSW_Sources})
set(Headers ${Headers};${GLSW_Headers})
source_group(GLSW FILES ${GLSW_Sources} ${GLSW_Headers})

add_library(${CMAKE_PROJECT_NAME} STATIC ${Sources} ${Headers})
