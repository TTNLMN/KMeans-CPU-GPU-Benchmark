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

#include "../utils/kmeans.hxx"

template <typename T>
class KMeans {
public:
    KMeans(int k, int max_iters) : k(k), max_iters(max_iters) {}
    virtual ~KMeans() = default;

    /**
     * @brief Fits the K-Means model to the data.
     *
     * @param data The data to cluster.
     */
    virtual void fit(const std::vector<std::vector<T>>& data) = 0;

    /**
     * @brief Predicts the cluster assignments for the data.
     *
     * @param data The data to assign.
     * @return std::vector<int> The cluster assignments.
     */
    virtual std::vector<int> predict(const std::vector<std::vector<T>>& data) = 0;

    /**
     * @brief Retrieves the centroids of the clusters.
     *
     * @return std::vector<std::vector<T>> The centroids.
     */
    virtual std::vector<std::vector<T>> getCentroids() const { return centroids; }

protected:
    int k;
    int max_iters;
    std::vector<std::vector<T>> centroids;

    /**
     * @brief Initializes centroids by randomly selecting k data points.
     *
     * @param data The data to initialize from.
     */
    void initializeRandomCentroids(const std::vector<std::vector<T>>& data) {
        if (data.size() < static_cast<size_t>(k)) {
            throw std::invalid_argument("Number of clusters (k) cannot exceed number of data points.");
        }

        centroids.clear();
        std::sample(data.begin(), data.end(), std::back_inserter(centroids),
                   k, std::mt19937{std::random_device{}()});
    }

    /**
     * @brief Initializes centroids using the K-Means++ algorithm.
     *
     * @param data The data to initialize from.
     */
    void initializeCentroids(const std::vector<std::vector<T>>& data) {
        if (data.size() < static_cast<size_t>(k)) {
            throw std::invalid_argument("Number of clusters (k) cannot exceed number of data points.");
        }

        auto start = std::chrono::high_resolution_clock::now();

        centroids.clear();
        std::mt19937 gen(std::random_device{}());

        // Step 1: Randomly select the first centroid
        std::uniform_int_distribution<> dis(0, data.size() - 1);
        centroids.push_back(data[dis(gen)]);

        // Step 2: Select the remaining centroids
        for (int i = 1; i < k; ++i) {
            std::vector<double> distances(data.size(), std::numeric_limits<double>::max());

            // Calculate the minimum distance of each point to the nearest centroid
            for (size_t j = 0; j < data.size(); ++j) {
                for (const auto& centroid : centroids) {
                    double dist = euclideanDistance(data[j], centroid);
                    if (dist < distances[j]) {
                        distances[j] = dist;
                    }
                }
            }

            // Step 3: Compute probabilities proportional to the squared distances
            double sum = std::accumulate(distances.begin(), distances.end(), 0.0);
            std::vector<double> probabilities(data.size());

            for (size_t j = 0; j < distances.size(); ++j) {
                probabilities[j] = distances[j] / sum;
            }

            // Step 4: Choose the next centroid based on the probability distribution
            std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
            centroids.push_back(data[dist(gen)]);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "Centroid initialization completed in " << elapsed.count() << " ms." << std::endl;
    }

    /**
     * @brief Finds the index of the closest centroid to a given point.
     *
     * @param point The data point.
     * @return int The index of the closest centroid.
     */
    int closestCentroid(const std::vector<T>& point) {
        int closest = 0;
        T min_distance = euclideanDistance(point, centroids[0]);

        for (int i = 1; i < k; ++i) {
            T distance = euclideanDistance(point, centroids[i]);
            if (distance < min_distance) {
                min_distance = distance;
                closest = i;
            }
        }
        return closest;
    }

    /**
     * @brief Updates the centroids based on current assignments.
     *
     * @param data The data points.
     * @param assignments The current cluster assignments.
     */
    void updateCentroids(const std::vector<std::vector<T>>& data, const std::vector<int>& assignments) {
        size_t dim = data[0].size();
        std::vector<std::vector<T>> new_centroids(k, std::vector<T>(dim, 0));
        std::vector<int> counts(k, 0);

        for (size_t i = 0; i < data.size(); ++i) {
            int cluster_id = assignments[i];
            for (size_t j = 0; j < dim; ++j) {
                new_centroids[cluster_id][j] += data[i][j];
            }
            counts[cluster_id]++;
        }

        for (int i = 0; i < k; ++i) {
            if (counts[i] > 0) {
                for (size_t j = 0; j < dim; ++j) {
                    new_centroids[i][j] /= counts[i];
                }
                centroids[i] = new_centroids[i];
            }
        }
    }
};