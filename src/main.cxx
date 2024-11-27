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

#ifndef DP
typedef double REAL;
#else
typedef float REAL;
#endif
#define check_out 1

#ifdef __linux__
#include <sstream>
#elif __APPLE__
#include <mach/mach.h>
#endif

// Function to get memory usage in KB
size_t getMemoryUsage() {
#ifdef __linux__
    std::ifstream stat_stream("/proc/self/status", std::ios_base::in);
    std::string line;
    size_t vmRSS = 0;
    while (std::getline(stat_stream, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string key;
            iss >> key >> vmRSS;
            break;
        }
    }
    return vmRSS;
#elif __APPLE__
    mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(),
                                 MACH_TASK_BASIC_INFO,
                                 reinterpret_cast<task_info_t>(&info),
                                 &infoCount);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Failed to get memory info: " << kr << std::endl;
        return EXIT_SUCCESS; // Could not retrieve memory info
    }
    return static_cast<size_t>(info.resident_size / 1024); // Convert bytes to KB
#else
    return EXIT_SUCCESS; // Unsupported platform
#endif
}

/*----------------------------------------------------------------------------*/
/* Toplevel function.                                                         */
/*----------------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    std::cout << "[K-Means Clustering Using CPU]" << std::endl;

    // Define parser
    args::ArgumentParser parser("K-Means Clustering Application", "Clusters data using K-Means algorithm.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<int> clustersFlag(parser, "clusters", "Number of clusters (k)", {'k', "clusters"}, 5);
    args::ValueFlag<std::string> executionFlag(parser, "execution", "Execution type: sequential or parallel", {'e', "execution"}, "sequential");
    args::ValueFlag<int> maxItersFlag(parser, "max_iters", "Maximum number of iterations", {'m', "max_iters"}, 100);
    args::ValueFlag<std::string> dataFolderFlag(parser, "folder", "The folder where to take the data", {'f', "folder"}, "synthetic");

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

    std::string inputPath = "../data/" + folder + "/data.csv";
    std::string outputPath = "../data/" + folder + "/labels.csv";

    std::cout << " Reading data from " << inputPath << std::endl;
    std::cout << " Writing labels to " << outputPath << std::endl;

    // Load data
    std::vector<std::vector<REAL>> data;
    try {
        if (folder.compare("synthetic") == 0) {
            io::CSVReader<3> in(inputPath);
            in.read_header(io::ignore_extra_column, "Feature1", "Feature2", "Feature3");
            REAL x, y, z;
            while (in.read_row(x, y, z)) {
                data.push_back({x, y, z});
            }
        } else if (folder.compare("pad") == 0) {
            io::CSVReader<3> in(inputPath);
            in.read_header(io::ignore_extra_column, "X", "Y", "Grey");
            REAL x, y;
            int grey;
            while (in.read_row(x, y, grey)) {
                // Since we want to cluster based on the greyscale value, we only keep the points that are grey
                if (grey == 1) data.push_back({x, y});
            }
        } else {
            std::cerr << "This folder has no emplementation" << std::endl;
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading data: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Define KMeans implementation
    std::unique_ptr<KMeans<REAL>> kmeans;

    if (executionType.compare("sequential") == 0) {
        kmeans = std::make_unique<KMeansSequential<REAL>>(k, max_iters);
    } else if (executionType.compare("parallel") == 0) {
        kmeans = std::make_unique<KMeansParallel<REAL>>(k, max_iters);
    } else {
        std::cerr << "Invalid execution type: " << executionType << ". Use 'sequential' or 'parallel'." << std::endl;
        return EXIT_FAILURE;
    }

    size_t memory_before = getMemoryUsage();

    auto start = std::chrono::system_clock::now();

    std::cout << " == Executing K-Means with " << k << " clusters using " << executionType << " execution." << std::endl;
    kmeans->fit(data);

    auto elapse = std::chrono::system_clock::now() - start;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapse);

    size_t memory_after = getMemoryUsage();

    /* Performance computation, results and performance printing ------------ */
    std::cout << " == Performances " << std::endl;
    std::cout << "\t Processing time: " << duration.count() << " (ms)" << std::endl;
    std::cout << "\t Memory Used: " << (memory_after - memory_before) << " KB" << std::endl;

    if (check_out) {
        std::vector<int> assignments = kmeans->predict(data);
        std::vector<std::vector<REAL>> centroids = kmeans->getCentroids();
        plotResults(outputPath, assignments, centroids);
    }
    
    return EXIT_SUCCESS;
}
