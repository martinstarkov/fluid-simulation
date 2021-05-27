cmake_minimum_required(VERSION 3.12)
project(fluid_simulation)

# Set C++ language requirement.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(SRC_FILES
"${SRC_DIR}/Application.cpp"
"${SRC_DIR}/FluidContainer.h"
"${SRC_DIR}/SYCLFluidContainer.h"
"${SRC_DIR}/FluidSimulation.h")

# Add include directories to project.

include_directories(
    ${SRC_DIR}
)

# Adding SYCL implementation.

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/cmake/")

set(ComputeCpp_DIR CACHE STRING "NOT-FOUND")

if (NOT ComputeCpp_DIR)
    message(FATAL_ERROR "SYCL implementation root not provided, please specify "
  "the path to the root of the chosen SYCL implementation using "
  "ComputeCpp_DIR=<path/to/install/root>.")
endif()

find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)

include_directories(SYSTEM "${ComputeCpp_INCLUDE_DIRS}")
include_directories(SYSTEM "${OpenCL_INCLUDE_DIR}")

# Create executable from all source files.

add_executable(
    ${PROJECT_NAME}
    ${SRC_FILES}
)

target_link_libraries (
  ${PROJECT_NAME}
  "${OpenCL_LIBRARIES}"
)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# Add Engine.

add_subdirectory("vendor/protegon")
add_protegon_to(${PROJECT_NAME})

# Add SYCL.

add_sycl_to_target(
  TARGET ${PROJECT_NAME}
  SOURCES "${SRC_DIR}/Application.cpp"
)