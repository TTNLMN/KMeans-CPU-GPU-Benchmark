#pragma once

#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

#include "point.cuh"
#include "kmeans_kernel.cuh"

/**
 * @brief KMeans class for clustering data points in D-dimensional space.
 * 
 * @tparam T Data type of the coordinates.
 * @tparam D Number of dimensions.
 */
template <typename T, int D>
class KMeans {
public:
    /**
     * @brief Constructor.
     * 
     * @param k         Number of clusters.
     * @param max_iters Maximum number of iterations.
     */
    __host__ KMeans(int k, int max_iters)
        : k(k), max_iters(max_iters)
    {
        centroids = new Point<T, D>[k];
    }

    /**
     * @brief Destructor.
     */
    __host__ ~KMeans() {
        delete[] centroids;
    }

    /**
     * @brief Fits the KMeans model to the data.
     * 
     * @param data The data to fit the model to.
     * @param M    The number of data points.
     */
    __host__ void fit(Point<T, D>* data, size_t M) {
        // Allocate device memory for data and centroids
        Point<T, D>* data_d;
        Point<T, D>* centroids_d;
        cudaMalloc((void**)&data_d,     M * sizeof(Point<T, D>));
        cudaMalloc((void**)&centroids_d, k * sizeof(Point<T, D>));

        // Copy data from host to device
        cudaMemcpy(data_d, data, M * sizeof(Point<T, D>), cudaMemcpyHostToDevice);

        // Initialize centroids then copy to device
        initialize_centroids(data, M);
        cudaMemcpy(centroids_d, centroids, k * sizeof(Point<T, D>), cudaMemcpyHostToDevice);

        // Allocate sums and counts on device
        T*   sums_d   = nullptr;
        int* counts_d = nullptr;
        cudaMalloc((void**)&sums_d,   k * D * sizeof(T));
        cudaMalloc((void**)&counts_d, k * sizeof(int));

        // Set kernel configuration
        int threadsPerBlock = 256;
        int blocksPerGrid   = (M - 1) / threadsPerBlock + 1;
        int blocksCentroids = (k - 1) / threadsPerBlock + 1;

        // Run K-Means iterations
        for (int i = 0; i < max_iters; ++i) {
            // Assign clusters
            assign_clusters<T, D><<<blocksPerGrid, threadsPerBlock>>>(data_d, centroids_d, k, M);
            cudaDeviceSynchronize();

            // Reset sums, counts
            cudaMemset(sums_d,   0, k * D * sizeof(T));
            cudaMemset(counts_d, 0, k * sizeof(int));

            size_t sharedMemSize = k * D * sizeof(T) + k * sizeof(int);

            // Update centroids
            update_centroids<T, D><<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(data_d, sums_d, counts_d, k, M);
            cudaDeviceSynchronize();

            // Compute new centroids
            compute_new_centroids<T, D><<<blocksCentroids, threadsPerBlock>>>(centroids_d, sums_d, counts_d, k);
            cudaDeviceSynchronize();
        }

        // Copy final centroids back to host
        cudaMemcpy(centroids, centroids_d, k * sizeof(Point<T, D>), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(data_d);
        cudaFree(centroids_d);
        cudaFree(sums_d);
        cudaFree(counts_d);
    }

    /**
     * @brief Predicts the cluster for each data point.
     * 
     * @param data The data to predict on.
     * @param M    The number of data points.
     * 
     * @return int* The cluster assignments.
     */
    __host__ int* predict(Point<T, D>* data, size_t M) {
        // Allocate device memory for the data
        Point<T, D>* data_d = nullptr;
        cudaMalloc((void**)&data_d, M * sizeof(Point<T, D>));
        cudaMemcpy(data_d, data, M * sizeof(Point<T, D>), cudaMemcpyHostToDevice);
        
        // Allocate device memory for the centroids
        // (We assume that the final centroids are stored on host in "centroids" after calling fit)
        Point<T, D>* centroids_d = nullptr;
        cudaMalloc((void**)&centroids_d, k * sizeof(Point<T, D>));
        cudaMemcpy(centroids_d, centroids, k * sizeof(Point<T, D>), cudaMemcpyHostToDevice);
        
        // Launch the assign_clusters kernel to compute the cluster for each data point.
        int threadsPerBlock = 256;
        int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;
        assign_clusters<T, D><<<blocksPerGrid, threadsPerBlock>>>(data_d, centroids_d, k, M);
        cudaDeviceSynchronize();
        
        // Copy the updated data back to host.
        Point<T, D>* result = new Point<T, D>[M];
        cudaMemcpy(result, data_d, M * sizeof(Point<T, D>), cudaMemcpyDeviceToHost);
        
        // Extract the cluster assignments into a separate array.
        int* assignments = new int[M];
        for (size_t i = 0; i < M; ++i) {
            assignments[i] = result[i].cluster;
        }
        
        // Clean up
        delete[] result;
        cudaFree(data_d);
        cudaFree(centroids_d);
        
        return assignments;
    }

protected:
    /**
     * @brief Number of clusters.
     */
    int k;

    /**
     * @brief Maximum number of iterations.
     */
    int max_iters;

    /**
     * @brief Centroids of the clusters.
     */
    Point<T, D>* centroids;

    /**
     * @brief Initializes the centroids by selecting random data points.
     * 
     * @param data The data to initialize from.
     * @param M    The number of data points.
     */
    __host__ void initialize_centroids_randomly(Point<T, D>* data, size_t M) {
        std::vector<int> indices(M);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        for (int i = 0; i < k; ++i) {
            centroids[i] = data[indices[i]];
        }
    }

    /**
     * @brief Initializes the centroids deterministically by selecting evenly spaced data points.
     * 
     * @param data The data to initialize from.
     * @param M    The number of data points.
     */
    __host__ void initialize_centroids(Point<T, D>* data, size_t M) {
        size_t stride = M / k;

        for (int i = 0; i < k; ++i) {
            size_t index = i * stride;
            centroids[i] = data[index];
        }
    }
};