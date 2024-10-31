// -----------------------------------------------------------------------------
/**
 * * Name:       kmeans_sequential.hxx
 * * Purpose:    Provide a sequential implementation of K-Means Clustering
 * * History:    Your Name, 2024
 */
// -----------------------------------------------------------------------------

#pragma once

#include "kmeans.hxx"

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
     */
    KMeansSequential(int k, int max_iters) : KMeans<T>(k, max_iters) {}

    /**
     * @brief Fits the K-Means model to the data sequentially.
     *
     * @param data The data to cluster.
     */
    void fit(const std::vector<std::vector<T>>& data) override {
        this->initializeRandomCentroids(data);
        std::vector<std::vector<T>> previous_centroids;

        for (int iter = 0; iter < this->max_iters; ++iter) {
            // Step 1: Assign each point to the closest centroid
            std::vector<int> assignments(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                assignments[i] = this->closestCentroid(data[i]);
            }

            // Step 2: Update centroids based on the assignments
            previous_centroids = this->centroids;
            this->updateCentroids(data, assignments);

            bool converged = true;
            for (size_t c = 0; c < this->centroids.size(); ++c) {
                for (size_t d = 0; d < this->centroids[c].size(); ++d) {
                    if (std::abs(this->centroids[c][d] - previous_centroids[c][d]) > 1e-6) {
                        converged = false;
                        break;
                    }
                }
                if (!converged) break;
            }

            if (converged) {
                std::cout << "\t Converged after " << iter + 1 << " iterations." << std::endl;
                break;
            }
        }
    }

    /**
     * @brief Predicts the cluster assignments for the data sequentially.
     *
     * @param data The data to assign.
     * @return std::vector<int> The cluster assignments.
     */
    std::vector<int> predict(const std::vector<std::vector<T>>& data) override {
        std::vector<int> assignments(data.size());

        for (size_t i = 0; i < data.size(); ++i) {
            assignments[i] = this->closestCentroid(data[i]);
        }
        
        return assignments;
    }
};