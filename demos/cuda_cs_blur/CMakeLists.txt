set(EXEC_NAME cuda_cs_blur)

set(DATA_DIRECTORY ${CMAKE_SOURCE_DIR}/${EXEC_NAME}/data/)
add_definitions(-DDATA_DIRECTORY="${DATA_DIRECTORY}")

#---------------------------------------------------------------------

find_package(CUDA)

if (TRUE) #(NOT CUDA_FOUND)
  # Current NVCC version does not support cpp-11
  # TODO : link separately CUDA and cpp files
	message(WARNING "${EXEC_NAME} will be compiled without CUDA support.")

	file(GLOB_RECURSE Sources *.cc)
	file(GLOB_RECURSE Headers *.h)
  file(GLOB_RECURSE Shaders *.glsl)

  add_executable(${EXEC_NAME} ${Sources} ${Headers} ${Shaders})
	target_link_libraries(${EXEC_NAME} ${LINKED_LIBRARIES})

else()
  add_definitions(-DUSE_CUDA)
  set(CUDA_NVCC_FLAGS "-gencode;arch=compute_30,code=sm_30;")

  # CUDA dont like c++11 or -Werror
  # Note: c++11 should be enable by specifiying nvcc for non cu* file,
  #       and g++ for c, cpp files
  #set(CMAKE_CXX_FLAGS        "-std=gnu++98")
  #set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")

  file(GLOB Sources *.cc *.cu)
  file(GLOB Headers *.h  *.cuh)

  #file(GLOB SourcesCUDA *.cu)
  #file(GLOB HeadersCUDA *.cuh)

  cuda_add_executable(${EXEC_NAME} ${Sources} ${Headers})
  target_link_libraries(${EXEC_NAME} ${LINKED_LIBRARIES})
endif()

#---------------------------------------------------------------------
