// -----------------------------------------------------------------------------
/**
 * * Name:       main.cxx
 * * Purpose:    Driver for K-Means Clustering on CPU
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

// Include KMeans utils
#include "../utils/kmeans.hxx"

#ifdef DP
typedef double REAL;
#else
typedef float REAL;
#endif

#define check_out 1

/*----------------------------------------------------------------------------*/
/* Toplevel function.                                                         */
/*----------------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    std::cout << "[K-Means Clustering Using CPU]" << std::endl;
    
    std::string inputPath = "../data/raw/test_pad.csv";
    std::string outputPath = "../data/processed/labels.csv";

    std::cout << " Reading data from " << inputPath << std::endl;
    std::cout << " Writing labels to " << outputPath << std::endl;

    // Define parser
    args::ArgumentParser parser("K-Means Clustering Application", "Clusters data using K-Means algorithm.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<int> clustersFlag(parser, "clusters", "Number of clusters (k)", {'k', "clusters"}, 120);
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
    int D = 2;

    // Load data
    std::vector<Point<REAL>> data;
    try {
        io::CSVReader<3> in(inputPath);
        in.read_header(io::ignore_extra_column, "X", "Y", "Grey");
        REAL x, y;
        int grey;
        while (in.read_row(x, y, grey)) {
            // Since we want to cluster based on the greyscale value, we only keep the points that are grey
            if (grey == 1) {
                REAL coords[] = { x, y };
                data.emplace_back(coords, D);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading data: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    size_t M = data.size();

    // Define KMeans implementation
    std::unique_ptr<KMeans<REAL>> kmeans;

    if (executionType == "sequential") {
        kmeans = std::make_unique<KMeansSequential<REAL>>(k, max_iters, D);
    } else if (executionType == "parallel") {
        kmeans = std::make_unique<KMeansParallel<REAL>>(k, max_iters, D);
    } else {
        std::cerr << "Invalid execution type: " << executionType << ". Use 'sequential' or 'parallel'." << std::endl;
        return EXIT_FAILURE;
    }

    auto start = std::chrono::system_clock::now();

    std::cout << " == Executing K-Means with " << k << " clusters using " << executionType << " execution..." << std::endl;
    kmeans->fit(data.data(), M);

    auto elapse = std::chrono::system_clock::now() - start;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapse);

    /* Performance computation, results and performance printing ------------ */
    std::cout << " == Performances " << std::endl;
    std::cout << "\t Processing time: " << duration.count() << " (ms)" << std::endl;

    if (check_out) {
        int* assignments = kmeans->predict(data.data(), M);
        plotResults(outputPath, assignments, M);
    }
    
    return EXIT_SUCCESS;
}
