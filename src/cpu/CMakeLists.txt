# CPU-specific CMakeLists

# Create an executable for the CPU version:
add_executable(kmeans_cpu main.cxx)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kmeans_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()