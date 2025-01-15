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
template <typename T, int D>
class KMeansParallel : public KMeans<T, D> {
public:
    /**
     * @brief Constructor to initialize the number of clusters and maximum iterations.
     *
     * @param k Number of clusters.
     * @param max_iters Maximum number of iterations.
     * @param D Dimensionality of each data point.
     */
    KMeansParallel(int k, int max_iters) : KMeans<T, D>(k, max_iters) {}

    /**
     * @brief Fits the K-Means model to the data in parallel.
     *
     * @param data The data to cluster.
     * @param M Number of data points.
     */
    void fit(Point<T, D>* data, size_t M) override {
        this->initializeCentroids(data, M);
        
        Point<T, D>* previous_centroids = new Point<T, D>[this->k];
        for (int c = 0; c < this->k; ++c) {
            previous_centroids[c] = this->centroids[c];
        }

        for (int iter = 0; iter < this->max_iters; ++iter) {
            // Step 1: Assign each point to the closest centroid in parallel
            assignClusters(data, M);

            // Step 2: Update centroids based on the assignments in parallel
            updateCentroids(data, M);

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
    int* predict(Point<T, D>* data, size_t M) override {
        int* assignments = new int[M];
        #pragma omp parallel for schedule(static)
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
    void assignClusters(Point<T, D>* data, size_t M) {
        #pragma omp parallel for schedule(static)
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
    void updateCentroids(Point<T, D>* data, size_t M) {
        int k = this->k;

        // Reset centroids
        for (int c = 0; c < k; ++c) {
            std::fill(this->centroids[c].data, this->centroids[c].data + D, T(0));
        }
        std::vector<int> counts(k, 0);

        int num_threads = omp_get_max_threads();
        
        // Allocate per-thread local accumulators
        std::vector<std::vector<T>> thread_centroids_sum(num_threads, std::vector<T>(k * D, T(0)));
        std::vector<std::vector<int>> thread_counts(num_threads, std::vector<int>(k, 0));

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            T* local_centroids = thread_centroids_sum[thread_id].data();
            int* local_counts = thread_counts[thread_id].data();

            #pragma omp for nowait
            for (size_t i = 0; i < M; ++i) {
                int cluster = data[i].cluster;
                local_counts[cluster]++;
                for (int d = 0; d < D; ++d) {
                    local_centroids[cluster * D + d] += data[i].data[d];
                }
            }
        }
        
        // Combine per-thread accumulators into global centroids and counts
        for (int t = 0; t < num_threads; ++t) {
            for (int c = 0; c < k; ++c) {
                counts[c] += thread_counts[t][c];
            }

            for (int c = 0; c < k; ++c) {
                for (int d = 0; d < D; ++d) {
                    this->centroids[c].data[d] += thread_centroids_sum[t][c * D + d];
                }
            }
        }

        // Divide by counts to get the mean
        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                for (int d = 0; d < D; ++d) {
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
