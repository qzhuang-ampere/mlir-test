include_directories(${CMAKE_SOURCE_DIR}/include/Dialect/Kestrel/TransformOps)
include_directories(${CMAKE_BINARY_DIR}/include/Dialect/Kestrel/TransformOps)

add_subdirectory(Dialect)
add_subdirectory(kestrel-opt)