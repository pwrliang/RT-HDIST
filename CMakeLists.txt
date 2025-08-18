cmake_minimum_required (VERSION 3.8)

project ("RTHD" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit 11.8 REQUIRED)

#OPTIX
set(OPTIX_PATH "" CACHE PATH "Path to OptiX SDK (7.4.0 or higher)")
set(_optix_search_paths
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.5.0"
    "/usr/local/OptiX/7.4.0"
    "/usr/local/OptiX/7.5.0"
    "$ENV{HOME}/NVIDIA-OptiX-SDK-7.4.0"
)

if(NOT OPTIX_PATH)
    foreach(p ${_optix_search_paths})
        if(EXISTS "${p}/include/optix.h")
            set(OPTIX_PATH "${p}" CACHE PATH "Path to OptiX SDK" FORCE)
            message(STATUS "Found OptiX automatically at ${OPTIX_PATH}")
            break()
        endif()
    endforeach()
endif()

if(NOT EXISTS "${OPTIX_PATH}/include/optix.h")
    message(FATAL_ERROR
        "OptiX not found! Please set OPTIX_PATH manually:\n"
        "  cmake -DOPTIX_PATH=/path/to/OptiX ..")
endif()

file(READ "${OPTIX_PATH}/include/optix.h" OPTIX_HEADER_CONTENTS)
string(REGEX MATCH "#define OPTIX_VERSION[ \t]+([0-9]+)" _ ${OPTIX_HEADER_CONTENTS})
set(OPTIX_VERSION ${CMAKE_MATCH_1})

if(OPTIX_VERSION LESS 70400)
    message(FATAL_ERROR "OptiX 7.4.0 or higher required, found version ${OPTIX_VERSION}")
else()
    message(STATUS "OptiX version ${OPTIX_VERSION} OK (>=70400)")
endif()

add_library(optix_common INTERFACE)
target_include_directories(optix_common INTERFACE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_PATH}/include
)
target_link_libraries(optix_common INTERFACE
    CUDA::cudart
    CUDA::cuda_driver
)

add_subdirectory("libs")
add_subdirectory("demo")