#pragma once

#include <cuda_runtime.h>

/**
 * @brief Point class representing a point in D-dimensional space.
 * 
 * @tparam T Data type of the coordinates.
 * @tparam D Number of dimensions.
 */
template <typename T, int D>
struct __align__(4 * sizeof(T)) Point {
    /**
     * @brief Coordinates of the point.
     * 
     * @note In the case of CUDA, we can't use a dynamic array here, so we use a fixed-size array.
     */
    T coords[D];

    // Cluster index (default: -1 indicates “unassigned”)
    int cluster;

    /**
     * @brief Default constructor.
     */
    __host__ __device__
    Point() : cluster(-1) {
        #pragma unroll
        for (int i = 0; i < D; i++) {
            coords[i] = T(0);
        }
    }

    /**
     * @brief Constructor with coordinates.
     * 
     * @param arr Array of coordinates.
     * @param cl Cluster index.
     */
    __host__ __device__
    Point(const T (&arr)[D], int cl = -1) : cluster(cl) {
        #pragma unroll
        for (int i = 0; i < D; i++) {
            coords[i] = arr[i];
        }
    }

    /**
     * @brief Computes the Euclidean distance between this Point and another.
     *
     * @param other Another Point.
     * @return T    The Euclidean distance.
     */
    __host__ __device__
    T distance(const Point<T, D>& other) const {
        T dist = 0;
        #pragma unroll
        for (int i = 0; i < D; i++) {
            T diff = coords[i] - other.coords[i];
            dist += diff * diff;
        }
        return dist;
    }
};