#
cmake_minimum_required (VERSION 3.8)

add_library(hausdorffLib
    ../../include/hausdorff/rtHausdorffQCluster.cpp
)
target_link_libraries(hausdorffLib PUBLIC
    optixBaseLib
    cudaLib
)
target_include_directories(hausdorffLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../
    ../CUDAUtil
    ../../include/hausdorff
)
