// -----------------------------------------------------------------------------
/**
 * * Name:       preprocess.cxx
 * * Purpose:    Preprocess the wine quality dataset
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

#ifdef DP
typedef double REAL;
#else
typedef float REAL;
#endif

int main() {
    std::cout << "[Data Preprocessing]" << std::endl;

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
    standardizeData(data);

    // Save preprocessed data to CSV
    std::string outputPath = "../data/preprocessed_data.csv";
    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        std::cerr << "Unable to open output file: " << outputPath << std::endl;
        return 1;
    }

    // Write header
    outFile << "fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,"
            << "free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol\n";

    // Write data
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            outFile << row[i];
            if (i != row.size() - 1) {
                outFile << ",";
            }
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "Preprocessed data saved to " << outputPath << std::endl;

    return 0;
}
