cmake_minimum_required(VERSION 3.31)
project(mj_rollout)

set(CMAKE_CXX_STANDARD 20)

# ------- Mujoco ------- #
include(FetchContent)
set(MUJOCO_VERSION "3.3.2" CACHE STRING "mujoco version")
set(COMP_ARCH "x86_64" CACHE STRING "computer architecture")

FetchContent_Declare(
        mujoco
        URL https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-${COMP_ARCH}.tar.gz
)

FetchContent_MakeAvailable(mujoco)

# ------- Setup the Mujoco library ------- #
message(STATUS "mujoco source dir: ${mujoco_SOURCE_DIR}")
find_library(MUJOCO_LIB mujoco REQUIRED HINTS ${mujoco_SOURCE_DIR}/lib)
add_library(mujoco-lib SHARED IMPORTED GLOBAL)
target_include_directories(mujoco-lib INTERFACE ${mujoco_SOURCE_DIR}/include)
target_include_directories(mujoco-lib INTERFACE ${mujoco_SOURCE_DIR}/include/mujoco)

set_property(TARGET mujoco-lib PROPERTY IMPORTED_LOCATION ${mujoco_SOURCE_DIR}/lib/libmujoco.so)


# Open MP
find_package(OpenMP REQUIRED)


# Rollout exec
add_executable(mj_rollout rollout.cpp poisson.cpp)
target_link_libraries(mj_rollout mujoco-lib OpenMP::OpenMP_CXX)
