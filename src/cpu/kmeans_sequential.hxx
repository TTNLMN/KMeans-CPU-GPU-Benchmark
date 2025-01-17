#pragma once

#include "kmeans.hxx"
#include <cstdlib>

/**
 * @brief Sequential implementation of K-Means clustering.
 *
 * @tparam T Numeric type of the data points (e.g., float, double).
 */
template <typename T>
class KMeansSequential : public KMeans<T> {
public:
    using Base = KMeans<T>;

    /**
     * @brief Constructor to initialize the number of clusters (k)
     *        and maximum iterations (max_iters).
     *
     * @param k          Number of clusters.
     * @param max_iters  Maximum number of iterations.
     */
    KMeansSequential(int k, int max_iters)
        : Base(k, max_iters)
    {}

    /**
     * @brief Fits the K-Means model to the data (sequentially).
     *
     * @param data Pointer to an array of Points.
     * @param M    Number of data points.
     */
    void fit(Point<T>* data, size_t M) override {
        this->initializeCentroids(data, M);

        // Prepare a container to track previous centroids
        std::vector<Point<T>> previous_centroids(this->k_);
        for (int c = 0; c < this->k_; ++c) {
            previous_centroids[c] = this->centroids_[c];
        }

        for (int iter = 0; iter < this->max_iters_; ++iter) {
            // Step 1: Assign each point to the closest centroid
            assignClusters(data, M);

            // Step 2: Update centroids based on assignments
            updateCentroids(data, M);

            // Step 3: Check for convergence
            bool converged = true;
            for (int c = 0; c < this->k_; ++c) {
                T change = this->centroids_[c].distance(previous_centroids[c]);
                if (change > static_cast<T>(1e-6)) {
                    converged = false;
                    break;
                }
            }
            if (converged) {
                std::cout << "\tConverged after " << iter + 1 << " iterations." << std::endl;
                break;
            }

            for (int c = 0; c < this->k_; ++c) {
                previous_centroids[c] = this->centroids_[c];
            }
        }
    }

    /**
     * @brief Predicts the cluster assignments for the data (sequentially).
     *
     * @param data Pointer to an array of Points.
     * @param M    Number of data points.
     * @return int* Dynamically allocated array of cluster assignments
     *              (caller is responsible for freeing).
     */
    int* predict(Point<T>* data, size_t M) override {
        int* assignments = new int[M];
        for (size_t i = 0; i < M; ++i) {
            assignments[i] = this->closestCentroid(data[i]);
        }
        return assignments;
    }

protected:
    /**
     * @brief Assigns each data point to the closest centroid.
     *
     * @param data Pointer to an array of Points.
     * @param M    Number of data points.
     */
    void assignClusters(Point<T>* data, size_t M) {
        for (size_t i = 0; i < M; ++i) {
            data[i].cluster = this->closestCentroid(data[i]);
        }
    }

    /**
     * @brief Updates the centroids by averaging all points assigned to each cluster.
     *
     * @param data Pointer to an array of Points.
     * @param M    Number of data points.
     */
    void updateCentroids(Point<T>* data, size_t M) {
        // Reset each centroid to 0
        for (int c = 0; c < this->k_; ++c) {
            std::fill(this->centroids_[c].coords.begin(), 
                      this->centroids_[c].coords.end(), T(0));
        }

        // Track how many points go into each cluster
        std::vector<int> counts(this->k_, 0);

        // Sum up all points in each cluster
        for (size_t i = 0; i < M; ++i) {
            int cluster_idx = data[i].cluster;
            counts[cluster_idx]++;
            // Add data[i]'s coordinates to the centroid
            auto& centroidCoords = this->centroids_[cluster_idx].coords;
            auto& pointCoords    = data[i].coords;
            if (centroidCoords.size() != pointCoords.size()) {
                throw std::runtime_error("Dimension mismatch in updateCentroids()");
            }
            for (size_t d = 0; d < pointCoords.size(); ++d) {
                centroidCoords[d] += pointCoords[d];
            }
        }

        // Compute the mean for each cluster
        for (int c = 0; c < this->k_; ++c) {
            if (counts[c] > 0) {
                auto& centroidCoords = this->centroids_[c].coords;
                for (size_t d = 0; d < centroidCoords.size(); ++d) {
                    centroidCoords[d] /= static_cast<T>(counts[c]);
                }
            } else {
                // If a cluster has no points, re-initialize that centroid randomly
                std::cout << "Cluster " << c << " has no points. "
                          << "Reinitializing centroid randomly." << std::endl;
                this->centroids_[c] = data[std::rand() % M];
            }
        }
    }
};