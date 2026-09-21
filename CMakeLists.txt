cmake_minimum_required(VERSION 3.14)
project(dohnuts-llama CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Pinned llama.cpp submodule (release tag, see .gitmodules).
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp" CACHE PATH "llama.cpp checkout")
if(NOT EXISTS "${LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp not found at ${LLAMA_DIR}. Run: git submodule update --init --depth 1")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp")

find_package(nlohmann_json 3.11 REQUIRED)

add_library(dohnuts-core
    src/engine.cpp
    src/protocol.cpp
)
target_include_directories(dohnuts-core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)
target_link_libraries(dohnuts-core PUBLIC llama nlohmann_json::nlohmann_json)

# HTTP transport adapted from lkarlslund/laya.cpp (MIT).
find_package(Threads REQUIRED)
add_library(dohnuts-http STATIC src/http.cpp)
target_include_directories(dohnuts-http PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_link_libraries(dohnuts-http PUBLIC nlohmann_json::nlohmann_json Threads::Threads)

add_executable(dohnuts-cli src/main.cpp)
target_link_libraries(dohnuts-cli PRIVATE dohnuts-core dohnuts-http)