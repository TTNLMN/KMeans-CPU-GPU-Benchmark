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

#include "args.hxx"
#include "csv.h"

#include "kmeans.hxx"
#include "kmeans_parallel.hxx"
#include "kmeans_sequential.hxx"

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

    // Define parser
    args::ArgumentParser parser("K-Means Clustering Application", "Clusters data using K-Means algorithm.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<int> clustersFlag(parser, "clusters", "Number of clusters (k)", {'k', "clusters"}, 120);
    args::ValueFlag<std::string> executionFlag(parser, "execution", "Execution type: sequential or parallel", {'e', "execution"}, "sequential");
    args::ValueFlag<int> maxItersFlag(parser, "max_iters", "Maximum number of iterations", {'m', "max_iters"}, 100);
    args::ValueFlag<std::string> dataFolderFlag(parser, "folder", "The folder where to take the data", {'f', "folder"}, "pad");

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
    std::string folder = args::get(dataFolderFlag);

    // Compose I/O paths
    std::string inputPath = "../data/" + folder + "/data.csv";
    std::string outputPath = "../data/" + folder + "/labels.csv";

    std::cout << " Reading data from " << inputPath << std::endl;
    std::cout << " Writing labels to " << outputPath << std::endl;

    // Load data
    std::vector<Point<REAL>> data;

    try {
        if (folder.compare("synthetic") == 0) {
            io::CSVReader<3> in(inputPath);
            in.read_header(io::ignore_extra_column, "Feature1", "Feature2", "Feature3");
            REAL x, y, z;
            while (in.read_row(x, y, z)) {
                std::vector<REAL> coords = { x, y, z };
                Point<REAL> pt;
                pt.coords = coords;
                data.push_back(pt);
            }
        } else if (folder.compare("pad") == 0) {
            io::CSVReader<3> in(inputPath);
            in.read_header(io::ignore_extra_column, "X", "Y", "Grey");
            REAL x, y;
            int grey;
            while (in.read_row(x, y, grey)) {
                // Since we want to cluster based on the greyscale value, we only keep the points that are grey
                if (grey == 1) {
                    std::vector<REAL> coords = { x, y };
                    Point<REAL> pt;
                    pt.coords = coords;
                    data.push_back(pt);
                }
            }
        } else {
            std::cerr << "This folder has no emplementation" << std::endl;
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading data: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    size_t M = data.size();
    if (M == 0) {
        std::cerr << "No data loaded. Exiting." << std::endl;
        return EXIT_FAILURE;
    }

    size_t dimension = data[0].coords.size();
    std::cout << " Loaded " << M << " data points of dimension " << dimension << "." << std::endl;

    // Define KMeans implementation
    std::unique_ptr<KMeans<REAL>> kmeans;

    if (executionType == "sequential") {
        kmeans = std::make_unique<KMeansSequential<REAL>>(k, max_iters);
    } else if (executionType == "parallel") {
        kmeans = std::make_unique<KMeansParallel<REAL>>(k, max_iters);
    } else {
        std::cerr << "Invalid execution type: " << executionType
                  << ". Use 'sequential' or 'parallel'." << std::endl;
        return EXIT_FAILURE;
    }

    auto start = std::chrono::system_clock::now();

    std::cout << " == Executing K-Means with " << k 
              << " clusters using " << executionType 
              << " execution..." << std::endl;

    kmeans->fit(data.data(), M);

    auto elapse = std::chrono::system_clock::now() - start;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapse);

    /* Performance computation, results and performance printing ------------ */
    std::cout << " == Performances " << std::endl;
    std::cout << "\t Processing time: " << duration.count() << " (ms)" << std::endl;

    if (check_out) {
        int* assignments = kmeans->predict(data.data(), M);
        plotResults(outputPath, assignments, M);
        delete[] assignments;
    }
    
    return EXIT_SUCCESS;
}
