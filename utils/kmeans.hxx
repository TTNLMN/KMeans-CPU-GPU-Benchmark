#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <Dense>
#include <vector>

/**
 * @brief Computes the Euclidean distance between two points.
 *
 * @tparam T The data type of the points (e.g., double, float).
 * @param a The first point.
 * @param b The second point.
 * @return T The Euclidean distance.
 *
 * @throws std::invalid_argument If the points have different dimensions.
 */
template <typename T>
T euclideanDistance(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Points must have the same dimensions.");
    }
    T sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        T diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/**
 * @brief Performs the PCA (Principal Component Analysis) on the given data.
 * 
 * @tparam T The data type of the points (e.g., double, float).
 * @param data The input data.
 * @param n_components The number of principal components to keep.
 * 
 * @return std::vector<std::vector<T>> The transformed data.
 */
template <typename T>
std::vector<std::vector<T>> performPCA(const std::vector<std::vector<T>>& data, int n_components) {
    size_t num_samples = data.size();
    size_t num_features = data[0].size();

    // Convert data to Eigen matrix
    Eigen::MatrixXd X(num_samples, num_features);
    for (size_t i = 0; i < num_samples; ++i)
        for (size_t j = 0; j < num_features; ++j)
            X(i, j) = data[i][j];

    // Compute covariance matrix
    Eigen::MatrixXd covariance = (X.transpose() * X) / (num_samples - 1);

    // Perform Eigen decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(covariance);
    Eigen::VectorXd eigen_values = eigen_solver.eigenvalues();
    Eigen::MatrixXd eigen_vectors = eigen_solver.eigenvectors();

    // Sort eigenvalues and corresponding eigenvectors in descending order
    std::vector<std::pair<T, Eigen::VectorXd>> eigen_pairs;
    for (size_t i = 0; i < eigen_values.size(); ++i) {
        eigen_pairs.push_back(std::make_pair(eigen_values(i), eigen_vectors.col(i)));
    }

    // Sort in descending order
    auto compare_pairs = [](const std::pair<T, Eigen::VectorXd>& a,
                            const std::pair<T, Eigen::VectorXd>& b) {
        return a.first > b.first;  // Sort by eigenvalue, descending
    };

    std::sort(eigen_pairs.begin(), eigen_pairs.end(), compare_pairs);

    // Select top 'n_components' eigenvectors
    Eigen::MatrixXd projection_matrix(num_features, n_components);
    for (int i = 0; i < n_components; ++i) {
        projection_matrix.col(i) = eigen_pairs[i].second;
    }

    // Project data onto principal components
    Eigen::MatrixXd transformed_data = X * projection_matrix;

    std::vector<std::vector<T>> result(num_samples, std::vector<T>(n_components));
    for (size_t i = 0; i < num_samples; ++i)
        for (int j = 0; j < n_components; ++j)
            result[i][j] = transformed_data(i, j);

    return result;
}

/**
 * @brief Plots the results of the K-Means clustering.
 * 
 * @tparam T The data type of the points (e.g., double, float).
 * @param assignments The cluster assignments.
 * @param centroids The cluster centroids.
 * 
 * @throws std::runtime_error If the output file cannot be opened.
 */
template <typename T>
void plotResults(std::string outputPath, std::vector<int> assignments, std::vector<std::vector<T>> centroids) {
    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
       throw std::runtime_error("Unable to open output file: " + outputPath);
    }

    outFile << "PointID,ClusterID\n";

    for (size_t i = 0; i < assignments.size(); ++i) {
        outFile << i << "," << assignments[i] << "\n";
    }

    outFile.close();
}