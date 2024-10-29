#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>

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
