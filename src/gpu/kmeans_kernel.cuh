#pragma once

#include <cuda_runtime.h>
#include "point.cuh"

// ------------------------------------------------------------------
// Closest centroid device function
// ------------------------------------------------------------------
template <typename T, int D>
__device__ int closest_centroid(const Point<T, D>& point, 
                                const Point<T, D>* centroids, 
                                int k) 
{
    int closest_cluster = 0;
    T min_distance = point.distance(centroids[0]);
    for (int c = 1; c < k; ++c) {
        T dist = point.distance(centroids[c]);
        if (dist < min_distance) {
            min_distance = dist;
            closest_cluster = c;
        }
    }
    return closest_cluster;
}

// ------------------------------------------------------------------
// 1) assign_clusters kernel
// ------------------------------------------------------------------
template <typename T, int D>
__global__ void assign_clusters(Point<T, D>* data_d, 
                                const Point<T, D>* centroids_d,
                                int k, 
                                size_t M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        data_d[i].cluster = closest_centroid<T, D>(data_d[i], centroids_d, k);
    }
}

// ------------------------------------------------------------------
// 2) update_centroids kernel
// ------------------------------------------------------------------
template <typename T, int D>
__global__ void update_centroids(Point<T, D>* data_d, 
                                 T* sums_d, 
                                 int* counts_d, 
                                 int k, 
                                 size_t M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        int cluster = data_d[i].cluster;
        // Accumulate coords in sums_d
        for (int d = 0; d < D; ++d) {
            atomicAdd(&sums_d[cluster * D + d], data_d[i].coords[d]);
        }
        atomicAdd(&counts_d[cluster], 1);
    }
}

// ------------------------------------------------------------------
// 3) compute_new_centroids kernel
// ------------------------------------------------------------------
template <typename T, int D>
__global__ void compute_new_centroids(Point<T, D>* centroids_d, 
                                      T* sums_d, 
                                      int* counts_d, 
                                      int k)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < k && counts_d[c] > 0) {
        for (int d = 0; d < D; ++d) {
            centroids_d[c].coords[d] = sums_d[c * D + d] / counts_d[c];
        }
    }
}

template <typename T, int D>
__global__ void update_centroids_reduction(Point<T, D>* data_d, 
                                           T* sums_d, 
                                           int* counts_d, 
                                           int k, 
                                           size_t M)
{
    extern __shared__ char s_buf[];
    T* s_sums = (T*)s_buf;
    int* s_counts = (int*)&s_sums[k * D];

    for(int idx = threadIdx.x; idx < k*D; idx += blockDim.x){
        s_sums[idx] = 0;
    }
    for(int idx = threadIdx.x; idx < k; idx += blockDim.x){
        s_counts[idx] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        int cluster = data_d[i].cluster;
        for (int d = 0; d < D; ++d) {
            atomicAdd(&s_sums[cluster * D + d], data_d[i].coords[d]);
        }
        atomicAdd(&s_counts[cluster], 1);
    }
    __syncthreads();

    if(threadIdx.x == 0) {
        for(int c = 0; c < k; c++){
            for(int d = 0; d < D; d++){
                atomicAdd(&sums_d[c * D + d], s_sums[c * D + d]);
            }
            atomicAdd(&counts_d[c], s_counts[c]);
        }
    }
}