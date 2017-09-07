set(EXEC_NAME ik_demo)

file(GLOB_RECURSE Sources *.cc)
file(GLOB_RECURSE Headers *.h)

set(DATA_DIRECTORY ${CMAKE_SOURCE_DIR}/${EXEC_NAME}/data)
add_definitions(-DDATA_DIRECTORY="${DATA_DIRECTORY}")
add_definitions(-DARMA_DONT_USE_WRAPPER)

add_executable(${EXEC_NAME} ${Sources} ${Headers})

find_package(Armadillo REQUIRED)

include_directories(${ARMADILLO_INCLUDE_DIR})

set(LINKED_LIBRARIES ${LINKED_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_link_libraries(${EXEC_NAME} ${LINKED_LIBRARIES})
