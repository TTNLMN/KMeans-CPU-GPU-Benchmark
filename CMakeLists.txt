cmake_minimum_required(VERSION 3.12)

# set the project name
project(KMeansClustering LANGUAGES CXX)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/utils
    ${PROJECT_SOURCE_DIR}/Eigen
)

# Option to use double precision
option(USE_DOUBLE "Use double precision (define DP flag)" ON)

if(USE_DOUBLE)
    add_definitions(-DDP)
    message(STATUS "Using double precision.")
else()
    message(STATUS "Using float precision.")
endif()

# Enable OpenMP
if(APPLE)
    message(STATUS "Configuring for macOS.")
    # Path to libomp installed via Homebrew
    set(LIBOMP_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
    set(LIBOMP_LIB_DIR "/opt/homebrew/opt/libomp/lib")

    # Add OpenMP flags manually
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${LIBOMP_INCLUDE_DIR}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${LIBOMP_LIB_DIR} -lomp")
    
    # Optionally, set runtime library path
    set(CMAKE_INSTALL_RPATH "${LIBOMP_LIB_DIR}")
endif()
if(LINUX)
    message(STATUS "Configuring for Linux platform.")
    # Find OpenMP package
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
        message(FATAL_ERROR "OpenMP not found. Please install OpenMP.")
    endif()
endif()


# Add source files for the main application
set(SOURCES_MAIN
    src/main.cxx
)

# Add source files for the preprocess script
set(SOURCES_PREPROCESS
    src/preprocess.cxx
)

# Add executable for the main application
add_executable(kmeans ${SOURCES_MAIN})

# Add executable for the preprocess script
add_executable(preprocess ${SOURCES_PREPROCESS})

# Link OpenMP libraries
if(LINUX)
    target_link_libraries(kmeans PUBLIC OpenMP::OpenMP_CXX)
endif()

if(APPLE)
    set_target_properties(kmeans PROPERTIES
        BUILD_RPATH "${LIBOMP_LIB_DIR}"
        INSTALL_RPATH "${LIBOMP_LIB_DIR}"
    )
endif()