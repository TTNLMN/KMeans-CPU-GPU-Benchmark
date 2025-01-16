#pragma once

#include <cuda_runtime.h>

/**
 * @brief Point class representing a point in D-dimensional space.
 * 
 * @tparam T Data type of the coordinates.
 * @tparam D Number of dimensions.
 */
template <typename T, int D>
struct Point {
    T coords[D];
    int cluster;

    __host__ __device__
    Point() : cluster(-1) {
        #pragma unroll
        for (int i = 0; i < D; i++) {
            coords[i] = T(0);
        }
    }

    __host__ __device__
    Point(const T (&arr)[D], int cl = -1) : cluster(cl) {
        #pragma unroll
        for (int i = 0; i < D; i++) {
            coords[i] = arr[i];
        }
    }

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