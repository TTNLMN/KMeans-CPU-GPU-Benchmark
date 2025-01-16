#pragma once

#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

#include "point.cuh"
#include "kmeans_kernel.cuh"

template <typename T, int D>
class KMeans {
public:
    __host__ KMeans(int k, int max_iters)
        : k(k), max_iters(max_iters)
    {
        // Allocate host centroids (just normal CPU memory)
        centroids = new Point<T, D>[k];
    }

    __host__ ~KMeans() {
        delete[] centroids;
    }

    __host__ void fit(Point<T, D>* data, size_t M) {
        // 1) Allocate device memory for data and centroids
        Point<T, D>* data_d;
        Point<T, D>* centroids_d;
        cudaMalloc((void**)&data_d,     M * sizeof(Point<T, D>));
        cudaMalloc((void**)&centroids_d, k * sizeof(Point<T, D>));

        // 2) Copy data from host to device
        cudaMemcpy(data_d, data, M * sizeof(Point<T, D>), cudaMemcpyHostToDevice);

        // 3) Initialize centroids (on host) then copy to device
        initialize_centroids(data, M);
        cudaMemcpy(centroids_d, centroids, k * sizeof(Point<T, D>), cudaMemcpyHostToDevice);

        // 4) Allocate sums and counts on device
        T*   sums_d   = nullptr;
        int* counts_d = nullptr;
        cudaMalloc((void**)&sums_d,   k * D * sizeof(T));
        cudaMalloc((void**)&counts_d, k * sizeof(int));

        // 5) Set kernel configuration
        int threadsPerBlock = 256;
        int blocksPerGrid   = (M + threadsPerBlock - 1) / threadsPerBlock;
        int blocksCentroids = (k + threadsPerBlock - 1) / threadsPerBlock;

        // 6) Run K-Means iterations
        for (int i = 0; i < max_iters; ++i) {
            // (a) Assign clusters
            assign_clusters<T, D><<<blocksPerGrid, threadsPerBlock>>>(data_d, centroids_d, k, M);
            cudaDeviceSynchronize();

            // (b) Reset sums, counts
            cudaMemset(sums_d,   0, k * D * sizeof(T));
            cudaMemset(counts_d, 0, k * sizeof(int));

            // (c) Update centroids
            update_centroids<T, D><<<blocksPerGrid, threadsPerBlock>>>(data_d, sums_d, counts_d, k, M);
            cudaDeviceSynchronize();

            // (d) Compute new centroids
            compute_new_centroids<T, D><<<blocksCentroids, threadsPerBlock>>>(centroids_d, sums_d, counts_d, k);
            cudaDeviceSynchronize();
        }

        // 7) Copy final centroids back to host
        cudaMemcpy(centroids, centroids_d, k * sizeof(Point<T, D>), cudaMemcpyDeviceToHost);

        // 8) (Optional) also copy data back if you want to see cluster assignments on host
        cudaMemcpy(data, data_d, M * sizeof(Point<T, D>), cudaMemcpyDeviceToHost);

        // 9) Cleanup
        cudaFree(data_d);
        cudaFree(centroids_d);
        cudaFree(sums_d);
        cudaFree(counts_d);
    }

    __host__ int* predict(Point<T, D>* data, size_t M) {
        // 1. Allocate device memory for the data
        Point<T, D>* data_d = nullptr;
        cudaMalloc((void**)&data_d, M * sizeof(Point<T, D>));
        cudaMemcpy(data_d, data, M * sizeof(Point<T, D>), cudaMemcpyHostToDevice);
        
        // 2. Allocate device memory for the centroids
        // (We assume that the final centroids are stored on host in "centroids")
        Point<T, D>* centroids_d = nullptr;
        cudaMalloc((void**)&centroids_d, k * sizeof(Point<T, D>));
        cudaMemcpy(centroids_d, centroids, k * sizeof(Point<T, D>), cudaMemcpyHostToDevice);
        
        // 3. Launch the assign_clusters kernel to compute the cluster for each data point.
        int threadsPerBlock = 256;
        int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;
        assign_clusters<T, D><<<blocksPerGrid, threadsPerBlock>>>(data_d, centroids_d, k, M);
        cudaDeviceSynchronize();
        
        // 4. Copy the updated data (with cluster assignments) back to host.
        Point<T, D>* result = new Point<T, D>[M];
        cudaMemcpy(result, data_d, M * sizeof(Point<T, D>), cudaMemcpyDeviceToHost);
        
        // 5. Extract the cluster assignments into a separate array.
        int* assignments = new int[M];
        for (size_t i = 0; i < M; ++i) {
            assignments[i] = result[i].cluster;
        }
        
        // 6. Clean up the temporary memory on device and host.
        delete[] result;
        cudaFree(data_d);
        cudaFree(centroids_d);
        
        // 7. Return the assignments array (which is allocated on host)
        return assignments;
    }

protected:
    int k;
    int max_iters;
    Point<T, D>* centroids;

    // Simple random initialization of centroids from sample of data
    __host__ void initialize_centroids(Point<T, D>* data, size_t M) {
        std::vector<int> indices(M);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        for (int i = 0; i < k; ++i) {
            centroids[i] = data[indices[i]];
        }
    }
};