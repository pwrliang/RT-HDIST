#
cmake_minimum_required (VERSION 3.8)

add_executable (RTHDdemo "demo.cpp")
set_target_properties(RTHDdemo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../output
)
target_link_libraries(RTHDdemo PRIVATE
    optixBaseLib
    geometryLib
    cudaLib
    hausdorffLib
)
target_include_directories(RTHDdemo PUBLIC
    ../../libs
    ../../libs/geometry
    ../../libs/CUDAUtil
    ../../libs/hausdorff
)