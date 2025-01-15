#pragma once

#include <cmath>
#include <stdexcept>
#include <algorithm>

// Template now takes the dimension (D) as a compile-time constant.
template <typename T, int D>
class Point {
public:
    // Inline storage for coordinates.
    T data[D];
    int cluster;

    // Default constructor initializes the coordinates to 0 and cluster to -1.
    Point() : cluster(-1) {
        for (int i = 0; i < D; ++i)
            data[i] = T(0);
    }

    // Constructor from a C-style array of length D.
    Point(const T (&data_in)[D], int cluster_val = -1) : cluster(cluster_val) {
        std::copy(data_in, data_in + D, data);
    }

    // Copy constructor.
    Point(const Point<T, D>& other) : cluster(other.cluster) {
        std::copy(other.data, other.data + D, data);
    }

    // Assignment operator.
    Point& operator=(const Point<T, D>& other) {
        if (this != &other) {
            cluster = other.cluster;
            std::copy(other.data, other.data + D, data);
        }
        return *this;
    }

    /**
     * @brief Computes the Euclidean distance between this point and another.
     *
     * @param other Another point.
     * @return T The Euclidean distance.
     */
    T distance(const Point<T, D>& other) const {
        T sum = 0;
        for (int i = 0; i < D; ++i) {
            T diff = data[i] - other.data[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    /**
     * @brief Adds the coordinates of another point to this point.
     *
     * @param other Another point.
     */
    void add(const Point<T, D>& other) {
        for (int i = 0; i < D; ++i) {
            data[i] += other.data[i];
        }
    }

    /**
     * @brief Divides the coordinates of this point by a scalar.
     *
     * @param scalar The divisor.
     */
    void divide(T scalar) {
        for (int i = 0; i < D; ++i) {
            data[i] /= scalar;
        }
    }
};
