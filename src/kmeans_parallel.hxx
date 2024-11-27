// -----------------------------------------------------------------------------
/**
 * * Name:       kmeans_parallel.hxx
 * * Purpose:    Provide a parallel implementation of K-Means Clustering
 * * History:    Your Name, 2024
 */
// -----------------------------------------------------------------------------

#pragma once

#include "kmeans.hxx"
#include <omp.h>

/**
 * @brief Parallel implementation of K-Means clustering using multi-threading.
 *
 * @tparam T The data type of the data points (e.g., double, float).
 */
template <typename T>
class KMeansParallel : public KMeans<T> {
public:
    /**
     * @brief Constructor to initialize the number of clusters and maximum iterations.
     *
     * @param k Number of clusters.
     * @param max_iters Maximum number of iterations.
     */
    KMeansParallel(int k, int max_iters) : KMeans<T>(k, max_iters) {}

    /**
     * @brief Fits the K-Means model to the data in parallel.
     *
     * @param data The data to cluster.
     */
    void fit(const std::vector<std::vector<T>>& data) override {
        this->initializeRandomCentroids(data);

        std::vector<std::vector<T>> previous_centroids;

        for (int iter = 0; iter < this->max_iters; ++iter) {
            // Step 1: Assign each point to the closest centroid in parallel
            std::vector<int> assignments(data.size(), 0);

            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < data.size(); ++i) {
                assignments[i] = this->closestCentroid(data[i]);
            }

            // Step 2: Update centroids based on the assignments in parallel
            previous_centroids = this->centroids;
            this->updateCentroids(data, assignments);

            bool converged = true;
            #pragma omp parallel for reduction(&:converged)
            for (size_t c = 0; c < this->centroids.size(); ++c) {
                for (size_t d = 0; d < this->centroids[c].size(); ++d) {
                    if (std::abs(this->centroids[c][d] - previous_centroids[c][d]) > 1e-6) {
                        converged = false;
                    }
                }
            }

            if (converged) {
                std::cout << "\t Converged after " << iter + 1 << " iterations." << std::endl;
                break;
            }
        }
    }

    /**
     * @brief Predicts the cluster assignments for the data in parallel.
     *
     * @param data The data to assign.
     * @return std::vector<int> The cluster assignments.
     */
    std::vector<int> predict(const std::vector<std::vector<T>>& data) override {
        std::vector<int> assignments(data.size(), 0);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < data.size(); ++i) {
            assignments[i] = this->closestCentroid(data[i]);
        }

        return assignments;
    }

private:
    /**
     * @brief Updates the centroids based on current assignments in parallel.
     *
     * @param data The data points.
     * @param assignments The current cluster assignments.
     */
    void updateCentroids(const std::vector<std::vector<T>>& data, const std::vector<int>& assignments) {
        size_t dim = data[0].size();
        std::vector<std::vector<T>> new_centroids(this->k, std::vector<T>(dim, 0));
        std::vector<int> counts(this->k, 0);

        #pragma omp parallel for reduction(+:new_centroids[:this->k][:dim], counts[:this->k])
        for (size_t i = 0; i < data.size(); ++i) {
            int cluster_id = assignments[i];
            for (size_t j = 0; j < dim; ++j) {
                new_centroids[cluster_id][j] += data[i][j];
            }
            counts[cluster_id]++;
        }

        // Update centroids by averaging
        #pragma omp parallel for
        for (int i = 0; i < this->k; ++i) {
            if (counts[i] > 0) {
                for (size_t j = 0; j < dim; ++j) {
                    new_centroids[i][j] /= counts[i];
                }
                this->centroids[i] = new_centroids[i];
            }
        }
    }
};
