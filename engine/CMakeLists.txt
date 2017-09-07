cmake_minimum_required(VERSION 2.8)

project(aer-engine)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

set(THIRDPARTIES ${CMAKE_SOURCE_DIR}/thirdparty)
set(AER_SOURCE_DIR aer)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${THIRDPARTIES}
  ${AER_SOURCE_DIR}
  ${GLEW_INCLUDE_DIRS}
)

add_subdirectory(aer)
