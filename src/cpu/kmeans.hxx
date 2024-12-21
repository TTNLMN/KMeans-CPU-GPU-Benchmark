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

template <typename T>
class Point {
public:
    int cluster;
    T* data;
    int D;

    Point() : D(0), cluster(-1), data(nullptr) {}

    Point(int D) : D(D), cluster(-1) {
        data = new T[D];
        std::fill(data, data + D, T(0));
    }

    Point(const T* data_in, int D) : D(D), cluster(-1) {
        data = new T[D];
        std::copy(data_in, data_in + D, data);
    }

    Point(const Point& other) : D(other.D), cluster(other.cluster) {
        data = new T[D];
        std::copy(other.data, other.data + D, data);
    }

    ~Point() {
        delete[] data;
    }

    Point& operator=(const Point& other) {
        if (this != &other) {
            if (D != other.D) {
                delete[] data;
                D = other.D;
                data = new T[D];
            }
            cluster = other.cluster;
            std::copy(other.data, other.data + D, data);
        }
        return *this;
    }

    T distance(const Point& other) const {
        if (D != other.D) {
            throw std::invalid_argument("Dimension mismatch");
        }
        T sum = 0;
        for (int i = 0; i < D; ++i) {
            T diff = data[i] - other.data[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    void add(const Point& other) {
        if (D != other.D) {
            throw std::invalid_argument("Dimension mismatch");
        }
        for (int i = 0; i < D; ++i) {
            data[i] += other.data[i];
        }
    }

    void divide(T scalar) {
        for (int i = 0; i < D; ++i) {
            data[i] /= scalar;
        }
    }
};

template <typename T>
class KMeans {
public:
    KMeans(int k, int max_iters, int D) : k(k), max_iters(max_iters), D(D) {
        centroids = new Point<T>[k];
        for (int i = 0; i < k; ++i) {
            centroids[i] = Point<T>(D);
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
    virtual void fit(Point<T>* data, size_t M) = 0;

    /**
     * @brief Predicts the cluster assignments for the data.
     *
     * @param data The data to assign.
     * @param M The number of data points.
     *
     * @return std::vector<int> The cluster assignments.
     */
    virtual int* predict(Point<T>* data, size_t M) = 0;

protected:
    int k;
    int max_iters;
    int D;
    Point<T>* centroids;

    /**
     * @brief Initializes centroids by randomly selecting k data points.
     *
     * @param data The data to initialize from.
     * @param M The number of data points.
     */
    void initializeCentroids(Point<T>* data, size_t M) {
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
    int closestCentroid(const Point<T>& point) {
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
