#pragma once

#include <vector>

/**
 * @brief Standardizes the data by subtracting the mean and dividing by the standard deviation.
 * 
 * @tparam T The data type of the data points (e.g., double, float).
 * @param data The data to standardize.
 */
template <typename T>
void standardizeData(std::vector<std::vector<T>>& data) {
    size_t num_features = data[0].size();
    size_t num_samples = data.size();

    std::vector<T> means(num_features, 0.0);
    std::vector<T> std_devs(num_features, 0.0);

    // Calculate means
    for (size_t j = 0; j < num_features; ++j) {
        for (size_t i = 0; i < num_samples; ++i) {
            means[j] += data[i][j];
        }
        means[j] /= num_samples;
    }

    // Calculate standard deviations
    for (size_t j = 0; j < num_features; ++j) {
        for (size_t i = 0; i < num_samples; ++i) {
            std_devs[j] += (data[i][j] - means[j]) * (data[i][j] - means[j]);
        }
        std_devs[j] = std::sqrt(std_devs[j] / num_samples);
    }

    // Standardize data
    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            if (std_devs[j] != 0) {
                data[i][j] = (data[i][j] - means[j]) / std_devs[j];
            }
        }
    }
}
