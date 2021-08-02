cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


project(test_ste)

include_directories(/usr/local/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin)

find_library(STELIB_LIB STELib HINTS /usr/local/lib/)

add_executable(test_random src/test_ste_random.cpp)
add_executable(test_output src/test_ste_output.cpp)

target_link_libraries(test_random PUBLIC ${STELIB_LIB})
target_link_libraries(test_output PUBLIC ${STELIB_LIB})
