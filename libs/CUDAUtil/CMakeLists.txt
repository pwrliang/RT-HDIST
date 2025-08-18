add_library(cudaLib
    ../../include/CUDAUtil/MortonUtils.cu
    ../../include/CUDAUtil/ReduceUtil.cu
)

target_link_libraries(cudaLib PUBLIC
    optix_common
)

target_include_directories(cudaLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../libs
    ../../include/CUDAUtil
)