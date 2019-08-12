# CMakeLists.txt ____________________________________________________________________________________________________________

cmake_minimum_required(VERSION 3.9)
set( CMAKE_VERBOSE_MAKEFILE ON)

#____________________________________________________________________________________________________________________________

project( Fleck)
enable_language( CXX) 
enable_language( CUDA)

#____________________________________________________________________________________________________________________________

set(ROOT_PATH "${CMAKE_SOURCE_DIR}/../../../")  
set(SOURCE_PATH "${ROOT_PATH}src/")
set(BUILD_PATH "${ROOT_PATH}delivery/${CMAKE_SYSTEM_NAME}/")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BUILD_PATH}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_PATH}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BUILD_PATH}/lib)
set(CMAKE_BINARY_DIR ${BUILD_PATH}/bin)  
set(CMAKE_INSTALL_PREFIX ${BUILD_PATH}/bin)

#____________________________________________________________________________________________________________________________

include_directories(  
)

add_library(particles STATIC 
	  ${SOURCE_PATH}console/particle.cu
	  ${SOURCE_PATH}console/particle.h 	  
	  ${SOURCE_PATH}console/thursters.cu
	  ${SOURCE_PATH}console/thursters.h 
 )


#____________________________________________________________________________________________________________________________

target_compile_features(particles PUBLIC cxx_std_17)
 
set_target_properties( particles PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
 
add_executable(particle_test ${SOURCE_PATH}console/test.cu)
target_include_directories(particle_test PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(particle_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(particle_test PRIVATE particles) 
 
#____________________________________________________________________________________________________________________________


