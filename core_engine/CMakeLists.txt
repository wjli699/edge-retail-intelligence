cmake_minimum_required(VERSION 3.16)
project(edge_retail_core_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CORE_ENGINE_BUILD_TESTS "Build core engine unit tests" OFF)

add_library(core_engine_lib
  src/app.cpp
)
target_include_directories(core_engine_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(edge_retail_core_engine
  src/main.cpp
)
target_link_libraries(edge_retail_core_engine PRIVATE core_engine_lib)

if(CORE_ENGINE_BUILD_TESTS)
  enable_testing()
  # add_subdirectory(tests) when tests exist
endif()
