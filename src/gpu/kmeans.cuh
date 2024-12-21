#pragma once

#include <vector>
#include <iostream>
#include <numeric>

#include <kmeans_kernel.cuh>

template <typename T, int D>
struct Point {
    T data[D];
    int cluster;

    __host__ __device__
    T distance(const Point<T, D>& other) const {
        T dist = 0;
        for (int i = 0; i < D; ++i) {
            T diff = data[i] - other.data[i];
            dist += diff * diff;
        }
        return dist;
    }
};

template <typename T, int D>
class KMeans {
public:
    __host__ KMeans(int k, int max_iters) : k(k), max_iters(max_iters) {
        centroids = new Point<T, D>[k];
    }

    __host__ void fit(Point<T, D>* data, size_t M) {
        // Allocate device memory and copy data
        Point<T, D>* data_d;
        Point<T, D>* centroids_d;

        cudaMalloc(&data_d, M * sizeof(Point<T, D>));
        cudaMalloc(&centroids_d, k * sizeof(Point<T, D>));

        cudaMemcpy(data_d, data, M * sizeof(Point<T, D>), cudaMemcpyHostToDevice);

        // Initialize centroids on host and copy to device
        initialize_centroids(data, M);
        cudaMemcpy(centroids_d, this->centroids, k * sizeof(Point<T, D>), cudaMemcpyHostToDevice);

        // Allocate sums and counts on device
        T* sums_d;
        int* counts_d;
        cudaMalloc(&sums_d, k * D * sizeof(T));
        cudaMalloc(&counts_d, k * sizeof(int));

        // Define threads and blocks
        int threadsPerBlock = 256;
        int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGridCentroids = (k + threadsPerBlock - 1) / threadsPerBlock;

        for (int i = 0; i < max_iters; ++i) {
            // Assign clusters
            assign_clusters<T, D><<<blocksPerGrid, threadsPerBlock>>>(data_d, centroids_d, k, M);

            // Reset sums and counts
            cudaMemset(sums_d, 0, k * D * sizeof(T));
            cudaMemset(counts_d, 0, k * sizeof(int));

            // Update centroids
            update_centroids<T, D><<<blocksPerGrid, threadsPerBlock>>>(data_d, sums_d, counts_d, k, M);

            // Compute new centroids
            compute_new_centroids<T, D><<<blocksPerGridCentroids, threadsPerBlock>>>(centroids_d, sums_d, counts_d, k);
        }

        // Copy centroids back to host
        cudaMemcpy(this->centroids, centroids_d, k * sizeof(Point<T, D>), cudaMemcpyDeviceToHost);

        // Copy data back to host if needed
        cudaMemcpy(data, data_d, M * sizeof(Point<T, D>), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(data_d);
        cudaFree(centroids_d);
        cudaFree(sums_d);
        cudaFree(counts_d);
    }

    __host__ int* predict(Point<T, D>* data, size_t M) {}

protected:
    int k;
    int max_iters;
    Point<T, D>* centroids;
    
    /**
     * @brief Initializes centroids by randomly selecting k data points.
     *
     * @param data The data to initialize from.
     * @param M The number of data points.
     */
    __host__ void initialize_centroids(Point<T, D>* data, size_t M) {
        std::vector<int> indices(M);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        for (int i = 0; i < k; ++i) {
            this->centroids[i] = data[indices[i]];
        }
    }

    /**
     * @brief Finds the index of the closest centroid to a given point.
     *
     * @param point The data point.
     * @param centroids_d The centroids to compare against.
     * @param k The number of centroids.
     * 
     * @return int The index of the closest centroid.
     */
    __device__ int closest_centroid(const Point<T, D>& point, Point<T, D>* centroids_d, int k) {
        int closest_cluster = 0;
        T min_distance = point.distance(centroids_d[0]);
        for (int c = 1; c < k; ++c) {
            T distance = point.distance(centroids_d[c]);
            if (distance < min_distance) {
                min_distance = distance;
                closest_cluster = c;
            }
        }
        return closest_cluster;
    }
};