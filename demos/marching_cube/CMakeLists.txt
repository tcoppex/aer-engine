set(EXEC_NAME marching_cube)

file(GLOB_RECURSE Sources *.cc)
file(GLOB_RECURSE Headers *.h)

set(APP_DATA_DIRECTORY ${CMAKE_SOURCE_DIR}/marching_cube/data/)
add_definitions(-DDATA_DIRECTORY="${APP_DATA_DIRECTORY}")

add_executable(${EXEC_NAME} ${Sources} ${Headers})
target_link_libraries(${EXEC_NAME} ${LINKED_LIBRARIES})
