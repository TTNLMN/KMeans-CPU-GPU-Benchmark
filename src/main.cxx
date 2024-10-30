// -----------------------------------------------------------------------------
/**
 * * Name:       main.cxx
 * * Purpose:    Driver for K-Means Clustering on PCA-Transformed Data
 * * History:    Titouan Le Moan & Max Bedel, 2024
 */
// -----------------------------------------------------------------------------

#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

// Include command-line argument parser
#include "args.hxx"

// Include the csv parser
#include "csv.h"

// Include KMeans implementations
#include "pca.hxx"
#include "kmeans.hxx"
#include "kmeans_parallel.hxx"
#include "kmeans_sequential.hxx"

#ifdef DP
typedef double REAL;
#else
typedef float REAL;
#endif
#define check_out 1

int main(int argc, char* argv[]) {
    std::cout << "[K-Means Clustering Application]" << std::endl;
    
    std::string inputPath = "../data/raw/winequality-red.csv";
    std::string outputPath = "../data/processed/labels.csv";

    std::cout << "Reading data from " << inputPath << std::endl;
    std::cout << "Writing labels to " << outputPath << std::endl;

    // Define parser
    args::ArgumentParser parser("K-Means Clustering Application", "Clusters data using K-Means algorithm.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<int> clustersFlag(parser, "clusters", "Number of clusters (k)", {'k', "clusters"}, 3);
    args::ValueFlag<std::string> executionFlag(parser, "execution", "Execution type: sequential or parallel", {'e', "execution"}, "sequential");
    args::ValueFlag<int> maxItersFlag(parser, "max_iters", "Maximum number of iterations", {'m', "max_iters"}, 100);

    // Parse command-line arguments
    try {
        parser.ParseCLI(argc, argv);
    } catch (const args::Help&) {
        std::cout << parser;
        return EXIT_SUCCESS;
    } catch (const args::ParseError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return EXIT_FAILURE;
    } catch (const args::ValidationError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return EXIT_FAILURE;
    }

    // Retrieve argument values
    int k = args::get(clustersFlag);
    std::string executionType = args::get(executionFlag);
    int max_iters = args::get(maxItersFlag);

    // Load data
    std::vector<std::vector<REAL>> data;
    try {
        io::CSVReader<11> in(inputPath);
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

    std::vector<std::vector<REAL>> transformed_data = performPCA(data, 2);

    // Choose KMeans implementation
    std::cout << "Executing K-Means with " << k << " clusters using " << executionType << " execution." << std::endl;

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // Define pointers for KMeans
    std::unique_ptr<KMeans<REAL>> kmeans;

    if (executionType == "sequential") {
        kmeans = std::make_unique<KMeansSequential<REAL>>(k, max_iters);
    } else if (executionType == "parallel") {
        kmeans = std::make_unique<KMeansParallel<REAL>>(k, max_iters);
    } else {
        std::cerr << "Invalid execution type: " << executionType << ". Use 'sequential' or 'parallel'." << std::endl;
        return EXIT_FAILURE;
    }

    // Fit the model
    kmeans->fit(transformed_data);

    // End timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "K-Means clustering completed in " << elapsed.count() << " ms." << std::endl;

    // Retrieve results
    std::vector<int> assignments = kmeans->predict(transformed_data);
    std::vector<std::vector<REAL>> centroids = kmeans->getCentroids();

    // Write results to CSV
    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        std::cerr << "Unable to open output file: " << outputPath << std::endl;
        return EXIT_FAILURE;
    }

    outFile << "PointID,ClusterID\n";

    for (size_t i = 0; i < assignments.size(); ++i) {
        outFile << i << "," << assignments[i] << "\n";
    }

    outFile.close();
    
    return EXIT_SUCCESS;
}
