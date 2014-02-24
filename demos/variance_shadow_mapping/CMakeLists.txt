set(EXEC_NAME variance_shadow_mapping)

file(GLOB_RECURSE Sources *.cc)
file(GLOB_RECURSE Headers *.h)

set(DATA_DIRECTORY ${CMAKE_SOURCE_DIR}/${EXEC_NAME}/data/)
add_definitions(-DDATA_DIRECTORY="${DATA_DIRECTORY}")

add_executable(${EXEC_NAME} ${Sources} ${Headers})
target_link_libraries(${EXEC_NAME} ${LINKED_LIBRARIES})
