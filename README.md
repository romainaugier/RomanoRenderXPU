# RomanoRenderXPU
XPU Pathtracer written in C++11 and OpenCL.

## Building

Clone with the submodules as RomanoRender heavily relies on stdromano, my C++11 standard library with a lot of utilities:
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

The only dependencies are, of course, a C++11 compiler, CMake, and GLFW and GLEW for OpenGL. As OpenCL is optional, it does not have to
be available. On Windows, it is recommended to install CUDA to get the OpenCL headers and library, and this is how the buildsystem
behaves for now. On Linux, install opencl-devel via your package manager.
Concerning GLFW and GLEW, I chose to use vcpkg to manage them. You can set it up like that:
```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh
```

```bat
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && bootstrap-vcpkg.bat
```

Then, you need to add the two dependencies:
```bash
vcpkg new --appplication
vcpkg add port glew
vcpkg add port glfw3
```

After that, use the build scripts depending on your platform. Various arguments can be passed:
- --debug: build in Debug mode
- --reldebug: build in RelWithDebInfo mode
- --clean: cleans the build directory
- --tests: runs the unit tests

Linux:
```bash
source setup_environment.sh
./build.sh --debug --tests
```

Windows:
```bat
call setup_environment.bat
build --debug --tests
```