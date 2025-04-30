cmake_minimum_required(VERSION 3.20)
# Enable C++ support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(mlir-test)

find_package(LLVM REQUIRED CONFIG HINTS ${LLVM_DIR})
find_package(MLIR REQUIRED CONFIG HINTS ${MLIR_DIR})
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(include)
add_subdirectory(lib)
add_library(my_library library.cpp)


add_executable(main main.cpp)

# Link the library to the main executable
target_link_libraries(main PRIVATE my_library )