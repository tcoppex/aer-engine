set(EXEC_NAME aura)

file(GLOB_RECURSE Sources *.cc)
file(GLOB_RECURSE Headers *.h)
file(GLOB_RECURSE Shaders *.glsl)

set(DATA_DIRECTORY ${CMAKE_SOURCE_DIR}/aura/data/)
add_definitions(-DDATA_DIRECTORY="${DATA_DIRECTORY}")

add_executable(${EXEC_NAME} ${Sources} ${Headers} ${Shaders})
target_link_libraries(${EXEC_NAME} ${LINKED_LIBRARIES})
