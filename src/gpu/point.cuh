#pragma once

#include <cuda_runtime.h>

/**
 * @brief Point class for K-means clustering
 * 
 * @tparam T Data type of the coordinates
 */
template <typename T>
struct Point {
    T* coords;
    int dimension;
    int cluster;

    __host__ __device__
    Point() 
        : coords(nullptr), dimension(0), cluster(-1) 
    {}

    __host__ __device__
    Point(T* arr, int dim, int cl = -1)
        : coords(arr), dimension(dim), cluster(cl)
    {}

    __host__ __device__
    T distance(const Point<T>& other) const {
        T dist = 0;
        for (int i = 0; i < dimension; i++) {
            T diff = coords[i] - other.coords[i];
            dist += diff * diff;
        }
        return dist;
    }
};
