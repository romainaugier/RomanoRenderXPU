#!/bin/bash

export VCPKG_ROOT=$PWD/vcpkg
export PATH=$VCPKG_ROOT:$PATH
export CMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
export ASAN_OPTIONS=new_delete_type_mismatch=0:protect_shadow_gap=0:force_dynamic_shadow=1