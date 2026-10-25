cmake_minimum_required(VERSION 3.20)
project(ptx C CXX)
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)  # FetchContent_Populate for header-only / prebuilt deps
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # compile_commands.json for clangd / IDE navigation
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Tune for the build machine (AVX2/FMA/BMI2 on Zen 3). Turn off to ship a binary that runs on any x86-64-v3 CPU.
option(PTX_NATIVE "Compile with -march=native" ON)
if(PTX_NATIVE)
  set(PTX_ARCH -march=native)
else()
  set(PTX_ARCH -march=x86-64-v3)  # AVX2 + FMA baseline
endif()

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(SDL3 REQUIRED CONFIG COMPONENTS SDL3)  # gamepads only (Steam Controller via HIDAPI); window stays GLFW
find_program(GLSLC glslc REQUIRED)

include(FetchContent)
FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.8
  GIT_SHALLOW TRUE)
FetchContent_Declare(cgltf
  GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
  GIT_TAG v1.15
  GIT_SHALLOW TRUE)
FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG a7d4c2175630e1ea5b6f8dd00ab96fed0b97b96d)
FetchContent_Declare(tracy
  GIT_REPOSITORY https://github.com/wolfpld/tracy.git
  GIT_TAG v0.14.1
  GIT_SHALLOW TRUE)
FetchContent_Declare(vma
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG v3.4.0
  GIT_SHALLOW TRUE)
FetchContent_Declare(meshoptimizer
  GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
  GIT_TAG v1.3
  GIT_SHALLOW TRUE)
FetchContent_Declare(entt
  GIT_REPOSITORY https://github.com/skypjack/entt.git
  GIT_TAG v4.0.0
  GIT_SHALLOW TRUE)
FetchContent_Declare(bc7enc
  GIT_REPOSITORY https://github.com/richgel999/bc7enc_rdo.git
  GIT_TAG b9438627eef73a1157e84201b6fa6eb2ffd6d9f0)
FetchContent_Declare(enkits
  GIT_REPOSITORY https://github.com/dougbinks/enkiTS.git
  GIT_TAG v1.12
  GIT_SHALLOW TRUE)
FetchContent_Declare(miniaudio
  GIT_REPOSITORY https://github.com/mackron/miniaudio.git
  GIT_TAG 0.11.22
  GIT_SHALLOW TRUE)
# Steam Audio: prebuilt SDK (ray-traced reflections / occlusion / transmission, HRTF)
FetchContent_Declare(steamaudio
  URL https://github.com/ValveSoftware/steam-audio/releases/download/v4.8.1/steamaudio_4.8.1.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(JoltPhysics
  GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
  GIT_TAG v5.5.0
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR Build)

# Jolt: SIMD up to AVX2 (no AVX-512 on Zen 3), no samples/tests, keep its asserts/profiler out of Release
set(USE_SSE4_1 ON)
set(USE_SSE4_2 ON)
set(USE_AVX ON)
set(USE_AVX2 ON)
set(USE_AVX512 OFF)
set(USE_LZCNT ON)
set(USE_TZCNT ON)
set(USE_F16C ON)
set(USE_FMADD ON)
set(TARGET_UNIT_TESTS OFF)
set(TARGET_HELLO_WORLD OFF)
set(TARGET_PERFORMANCE_TEST OFF)
set(TARGET_SAMPLES OFF)
set(TARGET_VIEWER OFF)
set(ENABLE_ALL_WARNINGS OFF)
set(INTERPROCEDURAL_OPTIMIZATION OFF)
set(CPP_RTTI_ENABLED ON)
set(DEBUG_RENDERER_IN_DEBUG_AND_RELEASE OFF)
set(PROFILER_IN_DEBUG_AND_RELEASE OFF)
set(GENERATE_DEBUG_SYMBOLS OFF)
set(FLOATING_POINT_EXCEPTIONS_ENABLED OFF)
set(ENABLE_INSTALL OFF)
set(JPH_ENABLE_VULKAN OFF)
# Tracy: off unless -DPTX_PROFILE=ON (zero cost when disabled)
option(PTX_PROFILE "Build with the Tracy profiler client" OFF)
set(TRACY_ENABLE ${PTX_PROFILE} CACHE BOOL "" FORCE)
set(TRACY_ON_DEMAND ON CACHE BOOL "" FORCE)
set(ENKITS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ENKITS_BUILD_C_INTERFACE OFF CACHE BOOL "" FORCE)
set(MESHOPT_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(imgui cgltf stb JoltPhysics tracy meshoptimizer enkits)
FetchContent_Populate(vma)  # header-only; populate avoids its install/sample targets
FetchContent_Populate(entt)
FetchContent_Populate(miniaudio)
FetchContent_Populate(bc7enc)  # only its BC7 block encoder is built (below)
FetchContent_Populate(steamaudio)
set(PHONON_ROOT ${steamaudio_SOURCE_DIR})
if(EXISTS ${steamaudio_SOURCE_DIR}/steamaudio)
  set(PHONON_ROOT ${steamaudio_SOURCE_DIR}/steamaudio)
endif()
add_library(phonon SHARED IMPORTED)
set_target_properties(phonon PROPERTIES
  IMPORTED_LOCATION ${PHONON_ROOT}/lib/linux-x64/libphonon.so
  INTERFACE_INCLUDE_DIRECTORIES ${PHONON_ROOT}/include)

add_library(imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)
add_library(bc7enc STATIC ${bc7enc_SOURCE_DIR}/bc7enc.cpp)
target_include_directories(bc7enc PUBLIC ${bc7enc_SOURCE_DIR})
target_compile_options(bc7enc PRIVATE -O3 ${PTX_ARCH})

target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

# Shaders -> SPIR-V
set(SHADER_SRC ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_OUT ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT})
# entry points in shaders/passes/, includes in topic folders (core/, sky/, terrain/, water/, ...), included root-relative
file(GLOB_RECURSE SHADERS CONFIGURE_DEPENDS ${SHADER_SRC}/*.comp ${SHADER_SRC}/*.vert ${SHADER_SRC}/*.frag)
file(GLOB_RECURSE SHADER_INCLUDES CONFIGURE_DEPENDS ${SHADER_SRC}/*.glsl)
set(SPV_FILES)
foreach(s ${SHADERS})
  get_filename_component(n ${s} NAME)
  set(out ${SHADER_OUT}/${n}.spv)
  add_custom_command(OUTPUT ${out}
    COMMAND ${GLSLC} --target-env=vulkan1.3 -O -I ${SHADER_SRC} ${s} -o ${out}
    DEPENDS ${s} ${SHADER_INCLUDES}
    COMMENT "glslc ${n}")
  list(APPEND SPV_FILES ${out})
endforeach()
add_custom_target(shaders ALL DEPENDS ${SPV_FILES})

# Engine sources: every .cpp under src/ (feature folders, see README.md). Headers are included as "folder/file.h".
file(GLOB_RECURSE PTX_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_executable(ptx ${PTX_SOURCES})
target_include_directories(ptx PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(ptx PRIVATE ${cgltf_SOURCE_DIR} ${stb_SOURCE_DIR} ${vma_SOURCE_DIR}/include ${entt_SOURCE_DIR}/src ${miniaudio_SOURCE_DIR})
target_compile_options(ptx PRIVATE ${PTX_ARCH})
target_compile_options(Jolt PRIVATE ${PTX_ARCH})
target_link_libraries(ptx PRIVATE bc7enc imgui Jolt Tracy::TracyClient meshoptimizer enkiTS SDL3::SDL3 phonon)
set_target_properties(ptx PROPERTIES BUILD_RPATH ${PHONON_ROOT}/lib/linux-x64)
target_compile_definitions(ptx PRIVATE SHADER_DIR="${SHADER_OUT}/" WORLD_DIR="${CMAKE_SOURCE_DIR}/world/")
# NVTX (header-only) for Nsight Systems ranges: the CUDA toolkit's copy, else NVIDIA's repo
find_path(NVTX3_INCLUDE nvtx3/nvToolsExt.h PATHS /opt/cuda/include /usr/local/cuda/include /usr/include)
if(NOT NVTX3_INCLUDE)
  FetchContent_Declare(nvtx GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git GIT_TAG v3.1.0 GIT_SHALLOW TRUE)
  FetchContent_Populate(nvtx)
  set(NVTX3_INCLUDE ${nvtx_SOURCE_DIR}/c/include)
endif()
target_include_directories(ptx PRIVATE ${NVTX3_INCLUDE})
target_compile_definitions(ptx PRIVATE PTX_NVTX=1)
add_dependencies(ptx shaders)

# DLSS Ray Reconstruction: NVIDIA DLSS SDK fetched into external/dlss (binaries are git LFS). Feature .so is copied next to the binary.
set(DLSS_SDK ${CMAKE_SOURCE_DIR}/external/dlss)
set(DLSS_TAG v310.9.16)
option(PTX_DLSS "Build with DLSS Ray Reconstruction" ON)
if(PTX_DLSS AND NOT EXISTS ${DLSS_SDK}/lib/Linux_x86_64/libnvsdk_ngx.a)
  find_package(Git QUIET)
  if(GIT_FOUND)
    message(STATUS "DLSS: fetching NVIDIA/DLSS ${DLSS_TAG}")
    if(NOT EXISTS ${DLSS_SDK}/.git)
      execute_process(COMMAND ${GIT_EXECUTABLE} -c filter.lfs.smudge= -c filter.lfs.required=false
                        clone --depth 1 --branch ${DLSS_TAG} https://github.com/NVIDIA/DLSS.git ${DLSS_SDK}
                      RESULT_VARIABLE DLSS_RC)
    endif()
    execute_process(COMMAND ${GIT_EXECUTABLE} lfs install --local WORKING_DIRECTORY ${DLSS_SDK} OUTPUT_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} lfs pull --include "include/**,lib/Linux_x86_64/**"
                    WORKING_DIRECTORY ${DLSS_SDK} RESULT_VARIABLE DLSS_RC)
    if(NOT DLSS_RC EQUAL 0)
      message(WARNING "DLSS: fetch failed (is git-lfs installed?)")
    endif()
  endif()
endif()
if(PTX_DLSS AND EXISTS ${DLSS_SDK}/lib/Linux_x86_64/libnvsdk_ngx.a)
  target_compile_definitions(ptx PRIVATE PTX_DLSS)
  target_include_directories(ptx PRIVATE ${DLSS_SDK}/include)
  target_link_libraries(ptx PRIVATE ${DLSS_SDK}/lib/Linux_x86_64/libnvsdk_ngx.a ${CMAKE_DL_LIBS})
  file(GLOB DLSSD_LIB ${DLSS_SDK}/lib/Linux_x86_64/rel/libnvidia-ngx-dlssd.so.*)
  add_custom_command(TARGET ptx POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DLSSD_LIB} $<TARGET_FILE_DIR:ptx>/)
  message(STATUS "DLSS Ray Reconstruction: enabled")
else()
  message(STATUS "DLSS Ray Reconstruction: SDK not found, disabled")
endif()
