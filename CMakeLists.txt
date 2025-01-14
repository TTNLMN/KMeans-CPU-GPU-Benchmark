cmake_minimum_required(VERSION 3.12)

# Project definition
project(KMeansClustering LANGUAGES CXX)

# Specify C++ standard globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# -------------------------------------------------------------------------------
# 1. Use double precision or not
# -------------------------------------------------------------------------------
option(USE_DOUBLE "Use double precision (define DP flag)" ON)

if(USE_DOUBLE)
    add_compile_definitions(DP) 
    message(STATUS "Using double precision.")
else()
    message(STATUS "Using float precision.")
endif()

# -------------------------------------------------------------------------------
# 2. Global include directories
# -------------------------------------------------------------------------------
include_directories(
    ${PROJECT_SOURCE_DIR}/Eigen
    ${PROJECT_SOURCE_DIR}/utils
)

# -------------------------------------------------------------------------------
# 3. Find OpenMP
# -------------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "[MacOS] Searching for OpenMP...")
    # Mac-specific flags for Clang + Homebrew libomp
    set(LIBOMP_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
    set(LIBOMP_LIB_DIR "/opt/homebrew/opt/libomp/lib")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${LIBOMP_INCLUDE_DIR}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${LIBOMP_LIB_DIR} -lomp")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "[Linux] Searching for OpenMP...")
    find_package(OpenMP REQUIRED)
endif()

# -------------------------------------------------------------------------------
# 4. Find CUDA
# -------------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "[Linux] Searching for CUDA...")
    find_package(CUDA)
endif()

# -------------------------------------------------------------------------------
# 5. Subdirectories for CPU and GPU
# -------------------------------------------------------------------------------
add_subdirectory(src/cpu)
add_subdirectory(src/gpu)