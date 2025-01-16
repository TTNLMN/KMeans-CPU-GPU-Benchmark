#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

/**
 * @brief A class representing a point in a multi-dimensional space.
 * 
 * @tparam T Numeric type (e.g. float, double).
 */
template <typename T>
class Point {
public:
    // Stores the coordinates with dynamic size.
    std::vector<T> coords;
    
    // Cluster index (default: -1 indicates “unassigned”)
    int cluster;

    /**
     * @brief Default constructor.
     */
    Point() : cluster(-1) {}

    /**
     * @brief Constructor from a std::vector<T> of coordinates.
     *
     * @param coords_in The coordinates to be stored in the Point.
     * @param cluster_val The cluster index for this Point.
     */
    Point(const std::vector<T>& coords_in, int cluster_val = -1)
        : coords(coords_in),
          cluster(cluster_val) {
    }

    /**
     * @brief Copy constructor.
     */
    Point(const Point<T>& other)
        : coords(other.coords),
          cluster(other.cluster) {
    }

    /**
     * @brief Assignment operator.
     */
    Point& operator=(const Point<T>& other) {
        if (this != &other) {
            coords = other.coords;
            cluster = other.cluster;
        }
        return *this;
    }

    /**
     * @brief Computes the Euclidean distance between this Point and another.
     *
     * @param other Another Point.
     * @return T    The Euclidean distance.
     */
    T distance(const Point<T>& other) const {
        if (coords.size() != other.coords.size()) {
            throw std::runtime_error("Dimension mismatch in distance computation");
        }
        T sum = 0;
        for (size_t i = 0; i < coords.size(); i++) {
            T diff = coords[i] - other.coords[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    /**
     * @brief Adds the coordinates of another Point to this one.
     *
     * @param other Another Point.
     */
    void add(const Point<T>& other) {
        if (coords.size() != other.coords.size()) {
            throw std::runtime_error("Dimension mismatch in add operation");
        }
        for (size_t i = 0; i < coords.size(); i++) {
            coords[i] += other.coords[i];
        }
    }

    /**
     * @brief Divides the coordinates of this Point by a scalar.
     *
     * @param scalar The divisor.
     */
    void divide(T scalar) {
        for (size_t i = 0; i < coords.size(); i++) {
            coords[i] /= scalar;
        }
    }
};
