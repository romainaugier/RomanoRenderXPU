@echo off

set VCPKG_ROOT=%CD%/vcpkg
set PATH=%VCPKG_ROOT%;%PATH%
set CMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
set ASAN_OPTIONS=new_delete_type_mismatch=0:force_dynamic_shadow=1