set(EXEC_NAME ik_demo)

file(GLOB_RECURSE Sources *.cc)
file(GLOB_RECURSE Headers *.h)

set(DATA_DIRECTORY ${CMAKE_SOURCE_DIR}/${EXEC_NAME}/data)
add_definitions(-DDATA_DIRECTORY="${DATA_DIRECTORY}")

find_package(Armadillo)
set(LINKED_LIBRARIES ${LINKED_LIBRARIES} ${ARMADILLO_LIBRARY})

add_executable(${EXEC_NAME} ${Sources} ${Headers})
target_link_libraries(${EXEC_NAME} ${LINKED_LIBRARIES})
