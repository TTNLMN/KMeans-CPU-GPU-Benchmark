#pragma once

template <typename T, int D>
__global__ void assign_clusters(Point<T, D>* data_d, const Point<T, D>* centroids_d, int k, size_t M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        data_d[i].cluster = closest_centroid<T, D>(data_d[i], centroids_d, k);
    }
}

template <typename T, int D>
__global__ void update_centroids(Point<T, D>* data_d, T* sums_d, int* counts_d, int k, size_t M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        int cluster = data_d[i].cluster;
        for (int d = 0; d < D; ++d) {
            atomicAdd(&sums_d[cluster * D + d], data_d[i].data[d]);
        }
        atomicAdd(&counts_d[cluster], 1);
    }
}

template <typename T, int D>
__global__ void compute_new_centroids(Point<T, D>* centroids_d, T* sums_d, int* counts_d, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < k && counts_d[c] > 0) {
        for (int d = 0; d < D; ++d) {
            centroids_d[c].data[d] = sums_d[c * D + d] / counts_d[c];
        }
    }
}
