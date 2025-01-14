// -----------------------------------------------------------------------------
/**
 * * Name:       kmeans_sequential.hxx
 * * Purpose:    Provide a sequential implementation of K-Means Clustering
 * * History:    Your Name, 2024
 */
// -----------------------------------------------------------------------------

#pragma once

#include "kmeans.hxx"
#include <cstdlib>

/**
 * @brief Sequential implementation of K-Means clustering.
 *
 * @tparam T The data type of the data points (e.g., double, float).
 */
template <typename T>
class KMeansSequential : public KMeans<T> {
public:
    /**
     * @brief Constructor to initialize the number of clusters and maximum iterations.
     *
     * @param k Number of clusters.
     * @param max_iters Maximum number of iterations.
     * @param D Dimensionality of each data point.
     */
    KMeansSequential(int k, int max_iters, int D) : KMeans<T>(k, max_iters, D) {}

    /**
     * @brief Fits the K-Means model to the data sequentially.
     *
     * @param data The data to cluster.
     * @param M Number of data points.
     */
    void fit(Point<T>* data, size_t M) override {
        this->initializeCentroids(data, M);

        Point<T>* previous_centroids = new Point<T>[this->k];
        for (int c = 0; c < this->k; ++c) {
            previous_centroids[c] = this->centroids[c];
        }

        for (int iter = 0; iter < this->max_iters; ++iter) {
            // Step 1: Assign each point to the closest centroid
            assignClusters(data, M);

            // Step 2: Update centroids based on the assignments
            updateCentroids(data, M);

            // Step 3: Check for convergence
            bool converged = true;
            for (int c = 0; c < this->k; ++c) {
                T change = this->centroids[c].distance(previous_centroids[c]);
                if (change > 1e-6) {
                    converged = false;
                    break;
                }
            }
            if (converged) {
                std::cout << "\t Converged after " << iter + 1 << " iterations." << std::endl;
                break;
            }

            // Update previous centroids
            for (int c = 0; c < this->k; ++c) {
                previous_centroids[c] = this->centroids[c];
            }
        }

        // Cleanup
        delete[] previous_centroids;
    }

    /**
     * @brief Predicts the cluster assignments for the data sequentially.
     *
     * @param data The data to assign.
     * @param M Number of data points.
     *
     * @return int* The cluster assignments.
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
     * @param data The data to cluster.
     * @param M Number of data points.
     */
    void assignClusters(Point<T>* data, size_t M) {
        for (size_t i = 0; i < M; ++i) {
            data[i].cluster = this->closestCentroid(data[i]);
        }
    }

    /**
     * @brief Updates the centroids based on current cluster assignments.
     *
     * @param data The data to cluster.
     * @param M Number of data points.
     */
    void updateCentroids(Point<T>* data, size_t M) {
        // Reset centroids
        for (int c = 0; c < this->k; ++c) {
            for (int d = 0; d < this->D; ++d) {
                this->centroids[c].data[d] = T(0);
            }
        }
        // Counts per cluster
        std::vector<int> counts(this->k, 0);

        // Sum up all points in each cluster
        for (size_t i = 0; i < M; ++i) {
            int cluster = data[i].cluster;
            counts[cluster]++;
            for (int d = 0; d < this->D; ++d) {
                this->centroids[cluster].data[d] += data[i].data[d];
            }
        }

        // Divide by counts to get the mean
        for (int c = 0; c < this->k; ++c) {
            if (counts[c] > 0) {
                for (int d = 0; d < this->D; ++d) {
                    this->centroids[c].data[d] /= static_cast<T>(counts[c]);
                }
            } else {
                // If a cluster has no points, reinitialize it randomly
                std::cout << "Cluster " << c << " has no points. Reinitializing centroid." << std::endl;
                this->centroids[c] = data[std::rand() % M];
            }
        }
    }
};