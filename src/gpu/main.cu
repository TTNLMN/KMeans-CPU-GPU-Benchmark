// -----------------------------------------------------------------------------
/**
 * * Name:       main.cu
 * * Purpose:    Driver for K-Means Clustering on GPU
 * * History:    Titouan Le Moan & Max Bedel, 2024
 */
// -----------------------------------------------------------------------------

#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

#include <cuda.h>

#include "args.hxx"
#include "csv.h"

#include "../utils/kmeans.hxx"

#include "kmeans.cuh"

#define check_out 1

#define REAL float
#define BLOCK_SIZE 32

/*----------------------------------------------------------------------------*/
/* Toplevel driver                                                            */
/*----------------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    std::cout << "[K-Means Clustering Using GPU]" << std::endl;
    
    // Define parser
    args::ArgumentParser parser("K-Means Clustering Application", "Clusters data using K-Means algorithm.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<int> clustersFlag(parser, "clusters", "Number of clusters (k)", {'k', "clusters"}, 120);
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
    int max_iters = args::get(maxItersFlag);
    std::string folder = args::get(dataFolderFlag);

    std::string inputPath = "../data/" + folder + "/data.csv";
    std::string outputPath = "../data/" + folder + "/labels.csv";

    std::cout << " Reading data from " << inputPath << std::endl;
    std::cout << " Writing labels to " << outputPath << std::endl;

    // Load data
    std::vector<Point<REAL>> data;
    try {
        if (folder.compare("pad") == 0) {
            io::CSVReader<3> in(inputPath);
            in.read_header(io::ignore_extra_column, "X", "Y", "Grey");
            REAL x, y;
            int grey;
            while (in.read_row(x, y, grey)) {
                // Since we want to cluster based on the greyscale value, we only keep the points that are grey
                if (grey == 1) {
                    REAL* coords = new REAL[2];
                    coords[0]    = x;
                    coords[1]    = y;

                    Point<REAL> p(coords, 2);
                    data.emplace_back(p);
                }
            }
        } else if (folder.compare("synthetic") == 0) {
            io::CSVReader<3> in(inputPath);
            in.read_header(io::ignore_extra_column, "Feature1", "Feature2", "Feature3");
            REAL x, y, z;
            while (in.read_row(x, y, z)) {
                REAL* coords = new REAL[3];
                coords[0]    = x;
                coords[1]    = y;
                coords[2]    = z;

                Point<REAL> p(coords, 3);
                data.emplace_back(p);
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

    // Setup CUDA environnement 
    cudaError_t error;

    cudaDeviceProp deviceProp;
    int devID = 0;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess) {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited) {
        std::cerr << "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice() ." <<std::endl;
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess) {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    } else {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // utilities
    cudaEvent_t start;
    cudaEvent_t stop;
    float msecTotal;

    // start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);

    KMeans<REAL> kmeans(k, max_iters);
    kmeans.fit(data.data(), M, data[0].dimension);
    
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    /* Performance computation, results and performance printing ------------ */
    std::cout << " == Performances " << std::endl;
    std::cout << "\t Processing time: " << msecTotal << " (ms)" << std::endl;

    if (check_out) {
        int* assignments = kmeans.predict(data.data(), M);
        plotResults(outputPath, assignments, M);
    }

    return EXIT_SUCCESS;
}