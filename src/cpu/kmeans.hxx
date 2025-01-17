#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <numeric>
#include "point.hxx"

/**
 * @brief Base class for K-Means clustering.
 * 
 * @tparam T Numeric type (float, double, etc.).
 */
template <typename T>
class KMeans {
public:
    /**
     * @brief Constructor for the KMeans base class.
     * 
     * @param k Number of clusters to create.
     * @param max_iters Maximum number of iterations allowed.
     */
    KMeans(int k, int max_iters)
        : k_(k), max_iters_(max_iters) {
        // Reserve space for centroids to avoid repeated allocations.
        centroids_.reserve(k_);
    }

    /**
     * @brief Virtual destructor.
     */
    virtual ~KMeans() = default;

    /**
     * @brief Pure virtual function to fit (train) the model on the data.
     * 
     * @param data Pointer to an array of Points (dynamic dimension).
     * @param M    Number of data points.
     */
    virtual void fit(Point<T>* data, size_t M) = 0;

    /**
     * @brief Pure virtual function to predict cluster assignments of new data.
     * 
     * @param data Pointer to an array of Points (dynamic dimension).
     * @param M    Number of data points.
     * @return int* Array of cluster assignments (caller must handle ownership).
     */
    virtual int* predict(Point<T>* data, size_t M) = 0;

protected:
    /**
     * @brief Number of clusters.
     */
    int k_;

    /**
     * @brief Maximum number of iterations allowed.
     */
    int max_iters_;

    /**
     * @brief Vector of centroid points, each with a dynamic dimension.
     */
    std::vector<Point<T>> centroids_;

    /**
     * @brief Randomly initializes centroids by selecting k distinct points.
     *
     * @param data The data to initialize from.
     * @param M    The number of data points.
     */
    void initializeCentroidsRandomly(Point<T>* data, size_t M) {
        centroids_.clear();
        centroids_.reserve(k_);

        std::vector<int> indices(M);
        std::iota(indices.begin(), indices.end(), 0);

        std::shuffle(indices.begin(), indices.end(),
                     std::mt19937{std::random_device{}()});

        for (int i = 0; i < k_; ++i) {
            centroids_.push_back(data[indices[i]]);
        }
    }

    /**
     * @brief Deterministically initializes centroids by selecting k points with a fixed stride.
     *
     * @param data The data to initialize from.
     * @param M    The number of data points.
     */
    void initializeCentroids(Point<T>* data, size_t M) {
        centroids_.clear();
        centroids_.reserve(k_);

        // Compute stride to evenly sample data points
        size_t stride = M / k_;

        for (size_t i = 0; i < k_; ++i) {
            size_t index = i * stride;
            centroids_.push_back(data[index]);
        }
    }

    /**
     * @brief Finds the index of the closest centroid to a given point.
     *
     * @param point A data point.
     * @return int  The index of the closest centroid in @c centroids_.
     */
    int closestCentroid(const Point<T>& point) const {
        if (centroids_.empty()) {
            // Technically should never happen if we call initializeCentroids first.
            throw std::runtime_error("Centroids have not been initialized.");
        }

        int closest_cluster = 0;
        T min_distance = point.distance(centroids_[0]);
        for (int c = 1; c < k_; ++c) {
            T distance_c = point.distance(centroids_[c]);
            if (distance_c < min_distance) {
                min_distance = distance_c;
                closest_cluster = c;
            }
        }
        return closest_cluster;
    }
};
