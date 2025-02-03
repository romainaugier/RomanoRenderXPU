@echo off

set VCPKG_ROOT=%CD%/vcpkg
set PATH=%VCPKG_ROOT%;%PATH%
set CMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake