// -----------------------------------------------------------------------------
/**
 * * Name:       preprocess.cxx
 * * Purpose:    Preprocess the wine quality dataset with PCA
 * * History:    Titouan Le Moan & Max Bedel, 2024
 */
// -----------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

// Include the csv parser
#include "csv.h"

// Include standardize function
#include "standardize.hxx"

// Include Eigen for PCA
#include <Dense>

#ifdef DP
typedef double REAL;
#else
typedef float REAL;
#endif

using namespace Eigen;

int main() {
    std::cout << "[Data Preprocessing with PCA]" << std::endl;

    // Load original data
    std::vector<std::vector<REAL>> data;
    try {
        io::CSVReader<11> in("../data/winequality-red.csv");
        in.read_header(io::ignore_extra_column, "fixed acidity", "volatile acidity", "citric acid",
                       "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                       "density", "pH", "sulphates", "alcohol");
        REAL fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
             free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol;
        while (in.read_row(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                           free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol)) {
            data.push_back({fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol});
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading data: " << e.what() << std::endl;
        return 1;
    }

    // Standardize data
    // standardizeData(data);

    // Convert data to Eigen matrix
    size_t num_samples = data.size();
    size_t num_features = data[0].size();
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
    std::vector<std::pair<REAL, Eigen::VectorXd>> eigen_pairs;
    for (size_t i = 0; i < eigen_values.size(); ++i) {
        eigen_pairs.emplace_back(eigen_values(i), eigen_vectors.col(i));
    }

    // Sort in descending order
    auto compare_pairs = [](const std::pair<REAL, Eigen::VectorXd>& a,
                            const std::pair<REAL, Eigen::VectorXd>& b) {
        return a.first > b.first;  // Sort by eigenvalue, descending
    };

    std::sort(eigen_pairs.begin(), eigen_pairs.end(), compare_pairs);

    int n_components = 2;
    Eigen::MatrixXd projection_matrix(num_features, n_components);
    for (int i = 0; i < n_components; ++i) {
        projection_matrix.col(i) = eigen_pairs[i].second;
    }

    // Project data onto principal components
    Eigen::MatrixXd transformed_data = X * projection_matrix;

    // Save PCA-transformed data to CSV
    std::string outputPath = "../data/preprocessed_data.csv";
    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        std::cerr << "Unable to open output file: " << outputPath << std::endl;
        return 1;
    }

    // Write header
    for (int i = 0; i < n_components; ++i) {
        outFile << "PC" << (i + 1);
        if (i != n_components - 1) {
            outFile << ",";
        }
    }
    outFile << "\n";

    // Write data
    for (int i = 0; i < transformed_data.rows(); ++i) {
        for (int j = 0; j < transformed_data.cols(); ++j) {
            outFile << transformed_data(i, j);
            if (j != transformed_data.cols() - 1) {
                outFile << ",";
            }
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "PCA-transformed data saved to " << outputPath << std::endl;

    return EXIT_SUCCESS;
}
