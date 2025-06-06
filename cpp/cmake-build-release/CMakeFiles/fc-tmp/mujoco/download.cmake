cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

message(VERBOSE "Executing download step for mujoco")

block(SCOPE_FOR VARIABLES)

include("/home/zolkin/AmberLab/Project-Rollout-Certifications/mj_rollout/cmake-build-release/CMakeFiles/fc-stamp/mujoco/download-mujoco.cmake")
include("/home/zolkin/AmberLab/Project-Rollout-Certifications/mj_rollout/cmake-build-release/CMakeFiles/fc-stamp/mujoco/verify-mujoco.cmake")
include("/home/zolkin/AmberLab/Project-Rollout-Certifications/mj_rollout/cmake-build-release/CMakeFiles/fc-stamp/mujoco/extract-mujoco.cmake")


endblock()
