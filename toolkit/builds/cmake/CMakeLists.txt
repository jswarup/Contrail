# CMakeLists.txt ____________________________________________________________________________________________________________


cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set( CMAKE_VERBOSE_MAKEFILE ON)

project(cmake_and_cuda LANGUAGES CXX CUDA)

#____________________________________________________________________________________________________________________________

set(ROOT_PATH "${CMAKE_SOURCE_DIR}/../../../")  
set(SOURCE_PATH "${ROOT_PATH}src/")

#____________________________________________________________________________________________________________________________

include(CTest)

add_library(particles STATIC 
	  ${SOURCE_PATH}console/particle.cu
	  ${SOURCE_PATH}console/particle.h 
 )

 #____________________________________________________________________________________________________________________________

target_compile_features(particles PUBLIC cxx_std_17)
 
set_target_properties( particles PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

if(BUILD_TESTING)
	add_executable(particle_test ${SOURCE_PATH}console/test.cu)

	set_target_properties(particle_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	target_link_libraries(particle_test PRIVATE particles)
	add_test(NAME particles_10k COMMAND particle_test 10000 )
	add_test(NAME particles_256k COMMAND particle_test 256000 )

endif()



