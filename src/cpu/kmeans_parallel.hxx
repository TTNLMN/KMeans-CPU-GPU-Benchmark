#pragma once

#include "kmeans.hxx"
#include <omp.h>
#include <cstdlib>

/**
 * @brief Parallel implementation of K-Means clustering using OpenMP.
 *
 * @tparam T Numeric type of the data points (e.g. float, double).
 */
template <typename T>
class KMeansParallel : public KMeans<T> {
public:
    using Base = KMeans<T>;

    /**
     * @brief Constructor to initialize the number of clusters (k) 
     *        and maximum iterations (max_iters).
     *
     * @param k Number of clusters.
     * @param max_iters Maximum number of iterations.
     */
    KMeansParallel(int k, int max_iters)
        : Base(k, max_iters)
    {}

    /**
     * @brief Fits the K-Means model to the data in parallel.
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
            // Step 1: Assign each point to the closest centroid (in parallel)
            assignClusters(data, M);

            // Step 2: Update centroids (in parallel)
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
     * @brief Predicts the cluster assignments for the data in parallel.
     *
     * @param data Pointer to an array of Points.
     * @param M    Number of data points.
     * @return int* Dynamically allocated array of cluster assignments
     *              (caller is responsible for freeing this array).
     */
    int* predict(Point<T>* data, size_t M) override {
        int* assignments = new int[M];
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i) {
            assignments[i] = this->closestCentroid(data[i]);
        }
        return assignments;
    }

protected:
    /**
     * @brief Assigns each data point to the closest centroid in parallel.
     *
     * @param data Pointer to an array of Points.
     * @param M    Number of data points.
     */
    void assignClusters(Point<T>* data, size_t M) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i) {
            data[i].cluster = this->closestCentroid(data[i]);
        }
    }

    /**
     * @brief Updates the centroids based on current cluster assignments, in parallel.
     *
     * @param data Pointer to an array of Points.
     * @param M    Number of data points.
     */
    void updateCentroids(Point<T>* data, size_t M) {
        int k = this->k_;
        if (k == 0) return;

        // For safety, check dimension from the first centroid
        size_t dimension = this->centroids_[0].coords.size();

        // Reset global centroids to 0
        for (int c = 0; c < k; ++c) {
            std::fill(this->centroids_[c].coords.begin(), 
                      this->centroids_[c].coords.end(), T(0));
        }

        // We'll need to track how many points go into each cluster
        // So we'll do this in parallel using thread local accumulators.
        std::vector<int> global_counts(k, 0);

        int num_threads = 1;
        #ifdef _OPENMP
        num_threads = omp_get_max_threads();
        #endif

        // Per thread local accumulators for sums and counts
        // Flatten each thread’s centroid accumulators into a single vector:
        // thread_centroids_sum[thread_id] has size k * dimension
        std::vector<std::vector<T>> thread_centroids_sum(
            num_threads, std::vector<T>(k * dimension, T(0))
        );
        std::vector<std::vector<int>> thread_counts(
            num_threads, std::vector<int>(k, 0)
        );

        // Parallel accumulation
        #pragma omp parallel
        {
            int thread_id = 0;
            #ifdef _OPENMP
            thread_id = omp_get_thread_num();
            #endif

            T* local_centroids = thread_centroids_sum[thread_id].data();
            int* local_counts   = thread_counts[thread_id].data();

            #pragma omp for nowait
            for (size_t i = 0; i < M; ++i) {
                int cluster_idx = data[i].cluster;
                local_counts[cluster_idx]++;

                // Add point[i]'s coordinates to the local centroid accumulator
                const auto& pointCoords = data[i].coords;
                if (pointCoords.size() != dimension) {
                    throw std::runtime_error(
                        "Dimension mismatch in updateCentroids()"
                    );
                }

                for (size_t d = 0; d < dimension; ++d) {
                    local_centroids[cluster_idx * dimension + d] += pointCoords[d];
                }
            }
        }

        // Combine per-thread accumulators into global centroids & counts
        for (int t = 0; t < num_threads; ++t) {
            // Combine counts
            for (int c = 0; c < k; ++c) {
                global_counts[c] += thread_counts[t][c];
            }

            // Combine sums
            const auto& thread_sum = thread_centroids_sum[t];
            for (int c = 0; c < k; ++c) {
                for (size_t d = 0; d < dimension; ++d) {
                    this->centroids_[c].coords[d] += 
                        thread_sum[c * dimension + d];
                }
            }
        }

        // Compute the mean for each centroid
        for (int c = 0; c < k; ++c) {
            if (global_counts[c] > 0) {
                for (size_t d = 0; d < dimension; ++d) {
                    this->centroids_[c].coords[d] /= 
                        static_cast<T>(global_counts[c]);
                }
            } else {
                // If a cluster has no points, re-initialize it randomly
                std::cout << "Cluster " << c 
                          << " has no points. Reinitializing centroid."
                          << std::endl;
                this->centroids_[c] = data[std::rand() % M];
            }
        }
    }
};
