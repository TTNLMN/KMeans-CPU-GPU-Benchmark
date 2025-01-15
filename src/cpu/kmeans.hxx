// -----------------------------------------------------------------------------
/**
 * * Name:       kmeans.hxx
 * * Purpose:    K-Means Clustering Implementation
 * * History:    Titouan Le Moan & Max Bedel, 2024
 */
// -----------------------------------------------------------------------------

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

template <typename T, int D>
class KMeans {
public:
    KMeans(int k, int max_iters) : k(k), max_iters(max_iters) {
        centroids = new Point<T, D>[k];
        for (int i = 0; i < k; ++i) {
            centroids[i] = Point<T, D>();
        }
    }

    virtual ~KMeans() {
        delete[] centroids;
    }

    /**
     * @brief Fits the K-Means model to the data.
     *
     * @param data The data to cluster.
     * @param M The number of data points.
     */
    virtual void fit(Point<T, D>* data, size_t M) = 0;

    /**
     * @brief Predicts the cluster assignments for the data.
     *
     * @param data The data to assign.
     * @param M The number of data points.
     *
     * @return std::vector<int> The cluster assignments.
     */
    virtual int* predict(Point<T, D>* data, size_t M) = 0;

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
    void initializeCentroids(Point<T, D>* data, size_t M) {
        std::vector<int> indices(M);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        for (int i = 0; i < k; ++i) {
            centroids[i] = data[indices[i]];
        }
    }

    /**
     * @brief Finds the index of the closest centroid to a given point.
     *
     * @param point The data point.
     * @return int The index of the closest centroid.
     */
    int closestCentroid(const Point<T, D>& point) {
        int closest_cluster = 0;
        T min_distance = point.distance(centroids[0]);
        for (int c = 1; c < k; ++c) {
            T distance = point.distance(centroids[c]);
            if (distance < min_distance) {
                min_distance = distance;
                closest_cluster = c;
            }
        }
        return closest_cluster;
    }
};
