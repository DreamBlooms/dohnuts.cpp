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
# mtmd provides the vision encoder used by --mmproj; builds standalone when
# LLAMA_BUILD_TOOLS is off.
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)

# Static linking produces a single self-contained binary; used for the Windows
# cross build. The default is shared.
option(DOHNUTS_STATIC "Link libraries statically" OFF)
if(DOHNUTS_STATIC)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
else()
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()

# Optional GPU backends. The default build is CPU only. Enable one or more to
# offload with --gpu-layers; see the README for per-backend requirements.
option(DOHNUTS_CUDA   "Build with the CUDA backend"   OFF)
option(DOHNUTS_VULKAN "Build with the Vulkan backend" OFF)
option(DOHNUTS_HIP    "Build with the ROCm/HIP backend" OFF)
option(DOHNUTS_METAL  "Build with the Metal backend"  OFF)
if(DOHNUTS_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(DOHNUTS_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(DOHNUTS_HIP)
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()
if(DOHNUTS_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

add_subdirectory("${LLAMA_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp")

# nlohmann/json is header-only and vendored under third_party/nlohmann so the
# build stays self-contained and cross-compilable (no host /usr/include leaks).
# Everything is compiled straight into one executable: the CLI is the only
# artifact, so no intermediate libraries are produced.

# The HTTP transport is adapted from lkarlslund/laya.cpp (MIT).
find_package(Threads REQUIRED)

add_executable(dohnuts-cli
    src/main.cpp
    src/engine.cpp
    src/protocol.cpp
    src/profile.cpp
    src/side.cpp
    src/side/runner.cpp
    src/side/decider.cpp
    src/side/kev.cpp
    src/http.cpp
)
target_include_directories(dohnuts-cli PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)
target_link_libraries(dohnuts-cli PRIVATE llama mtmd Threads::Threads)
if(WIN32)
    target_link_libraries(dohnuts-cli PRIVATE ws2_32)  # Winsock for cpp-httplib
endif()