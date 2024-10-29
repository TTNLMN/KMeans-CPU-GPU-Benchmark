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

    // Load PCA-transformed data
    std::vector<std::vector<REAL>> data;
    try {
        io::CSVReader<2> in("../data/preprocessed_data.csv");
        in.read_header(io::ignore_no_column, "PC1", "PC2");
        REAL pc1, pc2;
        while (in.read_row(pc1, pc2)) {
            data.push_back({pc1, pc2});
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading preprocessed data: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

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
    kmeans->fit(data);

    // End timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "K-Means clustering completed in " << elapsed.count() << " ms." << std::endl;

    // Retrieve results
    std::vector<int> assignments = kmeans->predict(data);
    std::vector<std::vector<REAL>> centroids = kmeans->getCentroids();

    // Write results to CSV
    std::string outputPath = "../plots/labels.csv";
    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        std::cerr << "Unable to open output file: " << outputPath << std::endl;
        return EXIT_FAILURE;
    }

    // Write header
    outFile << "PointID,ClusterID\n";

    for (size_t i = 0; i < assignments.size(); ++i) {
        outFile << i << "," << assignments[i] << "\n";
    }

    outFile.close();
    std::cout << "Cluster labels written to " << outputPath << std::endl;

    // Write centroids to CSV
    std::string centroidsPath = "../plots/centroids.csv";
    std::ofstream centroidFile(centroidsPath);
    if (centroidFile.is_open()) {
        // Write header
        centroidFile << "ClusterID";
        for (size_t dim = 0; dim < centroids[0].size(); ++dim) {
            centroidFile << ",Dim" << dim;
        }
        centroidFile << "\n";

        // Write centroids
        for (size_t i = 0; i < centroids.size(); ++i) {
            centroidFile << i;
            for (size_t j = 0; j < centroids[i].size(); ++j) {
                centroidFile << "," << centroids[i][j];
            }
            centroidFile << "\n";
        }

        centroidFile.close();
        std::cout << "Centroids written to " << centroidsPath << std::endl;
    } else {
        std::cerr << "Unable to open centroids file: " << centroidsPath << std::endl;
    }

    return EXIT_SUCCESS;
}
