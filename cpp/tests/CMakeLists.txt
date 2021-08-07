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
find_library(IBSLIB_LIB IBSLib HINTS /usr/local/lib/)

add_executable(test_random src/test_ste_random.cpp)
add_executable(test_output src/test_ste_output.cpp)
add_executable(test_longit src/test_ste_longitudinal.cpp)
add_executable(test_global src/test_ste_global.cpp)
add_executable(test_radiat src/test_ste_radiation.cpp)
add_executable(test_physic src/test_ste_phys.cpp)
add_executable(test_orig src/origcodetest.cpp)

# add_library(IBSLib  SHARED src/test_ste_longitudinal.cpp)
target_link_libraries(test_orig PUBLIC ${STELIB_LIB} ${IBSLIB_LIB})
target_link_libraries(test_random PUBLIC ${STELIB_LIB} ${IBSLIB_LIB})
target_link_libraries(test_output PUBLIC ${STELIB_LIB} )
target_link_libraries(test_longit PUBLIC ${STELIB_LIB} ${IBSLIB_LIB})
target_link_libraries(test_global PUBLIC ${STELIB_LIB} ${IBSLIB_LIB})
target_link_libraries(test_radiat PUBLIC ${STELIB_LIB} ${IBSLIB_LIB})
target_link_libraries(test_physic PUBLIC ${STELIB_LIB} ${IBSLIB_LIB})
