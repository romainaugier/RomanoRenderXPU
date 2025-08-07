# RomanoRenderXPU
XPU Pathtracer written in C++17 and CUDA.

## Building

Clone with the submodules as RomanoRender heavily relies on stdromano, my C++17 standard library with a lot of utilities, tinybvh and imnodes:
```bash
git clone --recurse-submodules https://github.com/romainaugier/RomanoRenderXPU.git
```

You then need to build stdromano, this should be straightforward:
```bash
cd stdromano
./build.sh
./build.sh --debug
```

```bat
cd stdromano
build
build --debug
```

Once stdromano is built, the headers and libs will be available in stdromano/install (default, and the CMake config points to this directory).

The dependencies are, of course, a C++17 compiler, CMake, and a few packages and Github repositories. CUDA is mandatory for now, install it system-wide and CMake should find it by itself, later it will be optional and deactivated if not available.
Concerning other dependencies (imgui, glfw...), I chose to use vcpkg to manage them. There is a custom fork tailored to the versions used by the renderer, along with custom packages for the ones not supported by default.
You can set it up like that:
```bash
git clone https://github.com/romainaugier/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh
```

```bat
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && bootstrap-vcpkg.bat
```

After that, use the build scripts depending on your platform, and Vcpkg will automatically build and install the packages needed. Various arguments can be passed:
- --debug: build in Debug mode
- --reldebug: build in RelWithDebInfo mode
- --clean: cleans the build directory
- --tests: runs the unit tests

```bash
source setup_environment.sh
./build.sh --debug --tests
```

```bat
call setup_environment.bat
build --debug --tests
```
