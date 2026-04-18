cmake_minimum_required(VERSION 3.16)
project(edge_retail_core_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# DeepStream SDK root — override with -DDS_SDK_PATH=... if installed elsewhere
set(DS_SDK_PATH "/opt/nvidia/deepstream/deepstream" CACHE PATH "DeepStream SDK root")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0)

find_package(yaml-cpp REQUIRED)

option(CORE_ENGINE_BUILD_TESTS "Build core engine unit tests" OFF)

set(CORE_ENGINE_SOURCES
  src/app.cpp
  src/pipeline.cpp
  src/metadata.cpp
)

add_library(core_engine_lib ${CORE_ENGINE_SOURCES})

target_include_directories(core_engine_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${DS_SDK_PATH}/sources/includes
  ${GST_INCLUDE_DIRS}
)

target_link_directories(core_engine_lib PUBLIC
  ${DS_SDK_PATH}/lib
)

target_link_libraries(core_engine_lib PUBLIC
  ${GST_LIBRARIES}
  nvdsgst_meta
  yaml-cpp
)

add_executable(edge_retail_core_engine src/main.cpp)
target_link_libraries(edge_retail_core_engine PRIVATE core_engine_lib)

if(CORE_ENGINE_BUILD_TESTS)
  enable_testing()
  # add_subdirectory(tests) when tests exist
endif()
