# MinGW-w64 cross toolchain for Windows x86_64.
#
#   cmake -S . -B build-win \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64-x86_64.cmake \
#     -DDOHNUTS_STATIC=ON
#
# Install the toolchain with: sudo apt-get install mingw-w64
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER   x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
set(CMAKE_RC_COMPILER  x86_64-w64-mingw32-windres)

set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Bundle the GCC runtime so the .exe runs without extra DLLs.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static -static-libgcc -static-libstdc++")
