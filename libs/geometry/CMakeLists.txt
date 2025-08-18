add_library(geometryLib
    ../../include/geometry/Object_t.cpp
    ../../include/geometry/PointCloud_impl.cpp
)
target_link_libraries(geometryLib PUBLIC
    optix_common
)
target_include_directories(geometryLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../libs
    ../../include/geometry
)
