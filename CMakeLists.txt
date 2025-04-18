cmake_minimum_required(VERSION 3.10)
# Enable C++ support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(mlir-test)

# Add the mlir-pass-extension subdirectory
# Add the library
add_library(my_library library.cpp)


add_executable(main main.cpp)

# Link the library to the main executable
target_link_libraries(main PRIVATE my_library)
