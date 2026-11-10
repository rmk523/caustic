cmake_minimum_required(VERSION 3.20)

set(CMAKE_BUILD_TYPE Release CACHE STRING "set build type to Release")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# --- Conda prefix ---
# 优先使用激活且确实装了 Boost 的环境；否则回退到 caustic_env。
# 这样即使 shell 里自动激活的是 base，也不会因为 CONDA_PREFIX 指错而找不到 Boost。
set(CAUSTIC_DEFAULT_PREFIX "/usr/local/miniconda3/envs/caustic_env")
set(CAUSTIC_CONDA_PREFIX "")

if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    file(GLOB _active_boost_cfg "$ENV{CONDA_PREFIX}/lib/cmake/Boost-*/BoostConfig.cmake")
    if(_active_boost_cfg)
        set(CAUSTIC_CONDA_PREFIX "$ENV{CONDA_PREFIX}")
        message(STATUS "CONDA_PREFIX = ${CAUSTIC_CONDA_PREFIX}")
    else()
        message(WARNING
            "当前激活环境 $ENV{CONDA_PREFIX} 中没有 Boost，"
            "改用 ${CAUSTIC_DEFAULT_PREFIX}")
    endif()
endif()

if(NOT CAUSTIC_CONDA_PREFIX)
    if(EXISTS "${CAUSTIC_DEFAULT_PREFIX}")
        set(CAUSTIC_CONDA_PREFIX "${CAUSTIC_DEFAULT_PREFIX}")
        message(WARNING "回退使用 ${CAUSTIC_CONDA_PREFIX}；建议先 conda activate caustic_env")
    else()
        message(FATAL_ERROR
            "找不到可用的 conda 环境，请先 conda activate caustic_env，"
            "或用 -DCAUSTIC_CONDA_PREFIX=/path/to/env 指定")
    endif()
endif()
list(PREPEND CMAKE_PREFIX_PATH "${CAUSTIC_CONDA_PREFIX}")

# --- CUDA toolkit（自动探测，不再写死 /usr/local/cuda-12.1） ---
# 优先级：-DCUDA_ROOT=... > -DCMAKE_CUDA_COMPILER=... > $CUDACXX > $CUDA_HOME > $CUDA_PATH > PATH 中的 nvcc
set(CUDA_ROOT "" CACHE PATH "CUDA toolkit root")

if(NOT CUDA_ROOT AND CMAKE_CUDA_COMPILER)
    get_filename_component(_caustic_nvcc_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(CUDA_ROOT "${_caustic_nvcc_dir}/.." ABSOLUTE)
endif()

if(NOT CUDA_ROOT)
    if(DEFINED ENV{CUDACXX} AND NOT "$ENV{CUDACXX}" STREQUAL "")
        get_filename_component(_caustic_nvcc_dir "$ENV{CUDACXX}" DIRECTORY)
        get_filename_component(CUDA_ROOT "${_caustic_nvcc_dir}/.." ABSOLUTE)
    elseif(DEFINED ENV{CUDA_HOME} AND NOT "$ENV{CUDA_HOME}" STREQUAL "")
        set(CUDA_ROOT "$ENV{CUDA_HOME}")
    elseif(DEFINED ENV{CUDA_PATH} AND NOT "$ENV{CUDA_PATH}" STREQUAL "")
        set(CUDA_ROOT "$ENV{CUDA_PATH}")
    else()
        find_program(_caustic_nvcc nvcc HINTS /usr/local/cuda/bin)
        if(NOT _caustic_nvcc)
            message(FATAL_ERROR
                "找不到 nvcc。请安装 CUDA toolkit，或用 -DCUDA_ROOT=/path/to/cuda 指定")
        endif()
        get_filename_component(_caustic_nvcc_dir "${_caustic_nvcc}" DIRECTORY)
        get_filename_component(CUDA_ROOT "${_caustic_nvcc_dir}/.." ABSOLUTE)
    endif()
endif()

if(NOT EXISTS "${CUDA_ROOT}/bin/nvcc")
    message(FATAL_ERROR "CUDA_ROOT=${CUDA_ROOT} 下没有 bin/nvcc")
endif()
message(STATUS "CUDA_ROOT = ${CUDA_ROOT}")

set(CMAKE_CUDA_COMPILER "${CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "CUDA compiler")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fpermissive")

# --- CUDA 架构 ---
# 默认读取当前 GPU 的 compute capability；可用 -DCAUSTIC_CUDA_ARCH=86 覆盖。
# 检测不到 GPU 时回退为常用的 sm_80/sm_86/sm_90 胖二进制。
set(CAUSTIC_CUDA_ARCH "" CACHE STRING "CUDA compute capability, e.g. 86")
if(NOT CAUSTIC_CUDA_ARCH)
    find_program(_caustic_nvidia_smi nvidia-smi)
    if(_caustic_nvidia_smi)
        execute_process(
            COMMAND "${_caustic_nvidia_smi}" --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _caustic_caps
            RESULT_VARIABLE _caustic_smi_rc
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_caustic_smi_rc EQUAL 0 AND _caustic_caps)
            string(REPLACE "\n" ";" _caustic_cap_list "${_caustic_caps}")
            list(GET _caustic_cap_list 0 _caustic_cap)
            string(REGEX REPLACE "[^0-9]" "" CAUSTIC_CUDA_ARCH "${_caustic_cap}")
        endif()
    endif()
endif()
if(NOT CAUSTIC_CUDA_ARCH)
    set(CAUSTIC_CUDA_ARCH "80;86;90" CACHE STRING "CUDA compute capability, e.g. 86" FORCE)
    message(WARNING "无法检测 GPU 架构，回退编译 sm_80/sm_86/sm_90")
endif()
set(CMAKE_CUDA_ARCHITECTURES "${CAUSTIC_CUDA_ARCH}")
message(STATUS "CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

project(caustic_GPU LANGUAGES CXX CUDA)

# --- Standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Boost (conda environment) ---
# 显式给出 include/lib 目录，并强制 module 模式：
# conda 的 BoostConfig.cmake 缺少 boost_headers 组件配置，config 模式会失败
set(BOOST_ROOT "${CAUSTIC_CONDA_PREFIX}")
set(BOOST_INCLUDEDIR "${CAUSTIC_CONDA_PREFIX}/include")
set(BOOST_LIBRARYDIR "${CAUSTIC_CONDA_PREFIX}/lib")
set(Boost_NO_SYSTEM_PATHS ON)
set(Boost_NO_BOOST_CMAKE ON)

find_package(Boost 1.78 REQUIRED)

message(STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")
message(STATUS "Boost_LIB_VERSION = ${Boost_LIB_VERSION}")

include_directories(${Boost_INCLUDE_DIRS})

# --- CGAL ---
find_package(CGAL 5.5 REQUIRED)

message(STATUS "CGAL FOUND")
message(STATUS "CGAL_DIR = ${CGAL_DIR}")
message(STATUS "CGAL_VERSION = ${CGAL_VERSION}")

# --- OpenMP ---
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message(STATUS "OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# --- nlohmann ---
find_package(nlohmann_json REQUIRED)

# --- files ---
file(GLOB SOURCES "./*.cpp" "./*.cu")
file(GLOB HEADERS "./*.h" "./*.hpp" "./*.cuh")

# --- executable ---
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# --- Eigen (conda environment, no longer external/eigen) ---
# conda 把 Eigen3Config.cmake 放在 share/eigen3/cmake；Linux 下大小写不匹配时
# 不会从 CMAKE_PREFIX_PATH 自动找到，这里显式给出
if(NOT Eigen3_DIR AND EXISTS "${CAUSTIC_CONDA_PREFIX}/share/eigen3/cmake/Eigen3Config.cmake")
    set(Eigen3_DIR "${CAUSTIC_CONDA_PREFIX}/share/eigen3/cmake")
endif()

find_package(Eigen3 3.3 QUIET NO_MODULE)
if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen3_DIR = ${Eigen3_DIR}")
    target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
else()
    # 回退：直接使用 conda 环境中的头文件目录
    set(EIGEN3_INCLUDE_DIR "${CAUSTIC_CONDA_PREFIX}/include/eigen3" CACHE PATH "Eigen include dir")
    if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
        message(FATAL_ERROR
            "找不到 Eigen3。请先 conda activate caustic_env，"
            "或通过 -DEIGEN3_INCLUDE_DIR=/path/to/eigen3 指定")
    endif()
    message(STATUS "Eigen3 include = ${EIGEN3_INCLUDE_DIR}")
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "${EIGEN3_INCLUDE_DIR}")
endif()

# --- CUDA toolkit libraries ---
set(CUDAToolkit_ROOT "${CUDA_ROOT}" CACHE PATH "CUDA toolkit root")
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit_VERSION = ${CUDAToolkit_VERSION}")

# --- cmake property ---
# CMAKE_CUDA_ARCHITECTURES 已在 project() 之前设置，这里不再单独覆盖

# --- CGAL ---
target_link_libraries(${PROJECT_NAME} CGAL::CGAL)

# --- cublas / curand ---
target_link_libraries(${PROJECT_NAME} CUDA::cublas CUDA::curand)

# --- Boost ---
if(Boost_LIBRARIES)
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()

# --- nlohmann ---
target_link_libraries(${PROJECT_NAME} nlohmann_json::nlohmann_json)

# Optimize code for CPU instruction set
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")