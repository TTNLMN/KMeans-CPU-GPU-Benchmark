# GPU-specific CMakeLists

# First check if CUDA was found in the top-level
if(CUDA_FOUND)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})

    add_executable(kmeans_gpu main.cu)

    target_link_libraries(kmeans_gpu PRIVATE ${CUDA_LIBRARIES})

else()
    message(WARNING "CUDA was not found; skipping kmeans_gpu build.")
endif()
