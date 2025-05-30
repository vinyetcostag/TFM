cmake_minimum_required(VERSION 3.10.0)

project(TuringPatternsMechanics VERSION 5.0)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${HPLFE_BASE_PATH}" CACHE PATH "Install directory" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

message(STATUS "Configuring ${CMAKE_PROJECT_NAME} project.")

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  message(STATUS "Hiperlife build type not set, default build type: ${CMAKE_BUILD_TYPE}")
else()
  message(STATUS "Hiperlife build type set to: ${CMAKE_BUILD_TYPE}")
endif()

find_package(hiperlife REQUIRED HINTS ${HL_BASE_PATH})

if(hiperlife_FOUND)
  message(STATUS "  hiperlife:\t Found version ${hiperlife_VERSION} at ${hiperlife_DIR}")
  message(STATUS "  hiperlife include dirs: ${hiperlife_INCLUDE_DIR}")
  message(STATUS "  hiperlife libs: ${hiperlife_LIBRARIES}")
endif(hiperlife_FOUND)

include_directories(SYSTEM "${Trilinos_INCLUDE_DIRS}")
include_directories(${hiperlife_INCLUDE_DIR})

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH ";${CMAKE_INSTALL_PREFIX}/lib;${hiperlife_LIBRARIES}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

## Convection-(reaction-)diffusion
set(hlConvectionDiffusion "hlConvectionDiffusion")
add_executable(${hlConvectionDiffusion} Physics.cpp ConvectionDiffusionProblem.cpp)

target_link_libraries(${hlConvectionDiffusion} ${Trilinos_LIBRARIES})
target_link_libraries(${hlConvectionDiffusion} ${hiperlife_LIBRARIES})
install(TARGETS ${hlConvectionDiffusion} DESTINATION ${PROJECT_INSTALL_PATH})

## Convection-(reaction-)diffusion with ALE
set(hlConvectionDiffusionALE "hlConvectionDiffusionALE")
add_executable(${hlConvectionDiffusionALE} Physics.cpp ConvectionDiffusionALEProblem.cpp)

target_link_libraries(${hlConvectionDiffusionALE} ${Trilinos_LIBRARIES})
target_link_libraries(${hlConvectionDiffusionALE} ${hiperlife_LIBRARIES})
install(TARGETS ${hlConvectionDiffusionALE} DESTINATION ${PROJECT_INSTALL_PATH})

