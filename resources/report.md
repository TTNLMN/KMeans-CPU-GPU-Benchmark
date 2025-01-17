## Abstract

This project aimed to benchmark the K-Means clustering algorithm in its sequential CPU implementation versus its parallelized GPU version using CUDA. K-Means, a foundational machine learning algorithm, often suffers from high computational demands on large datasets due to its iterative nature and reliance on distance calculations. By introducing parallelism with CUDA, we sought to accelerate the algorithm’s most computationally intensive steps—the assignment of points to clusters and the update of centroids.

Our solution involved implementing a sequential version in C++, a parallel version utilizing CUDA’s thread model, shared memory, and efficient memory access patterns, and an additional parallel version on the CPU using OpenMP directives. Experiments were conducted on a pad detection dataset with two dimensions, as well as a generated 3D dataset to enable visualization in three dimensions. These datasets were chosen to evaluate the algorithm's performance and flexibility across different dimensions.

The results demonstrated significant speedup on the GPU implementation, particularly on larger datasets, validating the potential of GPU parallelism in high-performance computing. However, memory management and thread synchronization posed notable challenges, leading to insights about optimizing CUDA applications. These findings underscore the value of parallelization in reducing computational overhead for iterative algorithms like K-Means.

## Design Methodology

### Overview of the Algorithm

The K-Means algorithm iteratively assigns data points to the nearest centroid (assignment step) and recalculates the centroids based on the mean of the assigned points (update step). These two steps repeat until convergence, i.e., when centroids no longer change significantly.

### Parallelization Opportunities

The computational bottlenecks in K-Means arise from:

1. Calculating the distance of each data point to all centroids.
2. Updating the centroids by aggregating and averaging the assigned points.

Both steps involve operations that can be parallelized:

- **Assignment Step**: Each data point’s closest centroid can be computed independently.
- **Update Step**: Reduction operations (summing assigned points) can be performed in parallel for each centroid.

### Master-Slave Model

#### Master

- Initializes centroids randomly.
- Manages global memory allocation and data transfer between CPU and GPU.
- Oversees termination conditions (e.g., maximum iterations; for simplicity, convergence checks were not implemented in the GPU version).

#### Slaves (CUDA Threads)

- Compute distances between points and centroids in parallel.
- Perform reduction operations using shared memory for centroid updates.

### Communication

- Data points and centroids are transferred from host to device before computation.
- Synchronization is critical within blocks to ensure all threads finish computations before proceeding.

### Optimizations

- **Shared Memory**: Used to store centroids locally within blocks to reduce global memory access. During the centroid update step, shared memory temporarily stored partial sums and counts, significantly reducing global memory access. This involved resetting sums and counts using `cudaMemset`, dynamically allocating shared memory, and performing atomic operations for reduction. Synchronization within blocks ensured consistent updates before transferring results back to global memory.
- **Thread Synchronization**: Managed with `__syncthreads()` to coordinate threads within a block.

The following pseudocode and explanations detail the key steps of the GPU implementation, emphasizing the thread hierarchy and memory access patterns:

1. **Memory Allocation and Initialization**:

   - Allocate global memory for data points, centroids, sums, and counts.
   - Copy the data points and initialized centroids from host to device memory.

   ```
   cudaMalloc(&data_d, M * sizeof(Point<T, D>));
   cudaMalloc(&centroids_d, k * sizeof(Point<T, D>));
   cudaMemcpy(data_d, data, M * sizeof(Point<T, D>), cudaMemcpyHostToDevice);
   cudaMemcpy(centroids_d, centroids, k * sizeof(Point<T, D>), cudaMemcpyHostToDevice);
   ```

2. **Cluster Assignment Kernel**:

   - Each thread computes the distance between its assigned data point and all centroids to find the closest cluster.

   ```
   assign_clusters<<<blocksPerGrid, threadsPerBlock>>>(data_d, centroids_d, k, M);
   ```

3. **Centroid Update Kernel with Shared Memory**:

   - Dynamically allocate shared memory for partial sums and counts.
   - Each block reduces contributions from its threads using atomic operations.
   - Synchronize threads within the block using `__syncthreads()`.

   ```
   update_centroids<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(data_d, sums_d, counts_d, k, M);
   ```

4. **Final Centroid Calculation**:

   - Update global centroids by averaging the sums and counts from the reduction.

   ```
   compute_new_centroids<<<blocksCentroids, threadsPerBlock>>>(centroids_d, sums_d, counts_d, k);
   ```

5. **Synchronization and Iteration**:

   - Synchronize device computations after each kernel call.
   - Repeat for a fixed number of iterations or until convergence.

   ```
   for (int i = 0; i < max_iters; ++i) {
       assign_clusters<<<blocksPerGrid, threadsPerBlock>>>(...);
       cudaMemset(...);
       update_centroids<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(...);
       compute_new_centroids<<<blocksCentroids, threadsPerBlock>>>(...);
   }
   ```

These pseudocode snippets demonstrate how the GPU implementation utilizes shared memory to optimize performance, ensuring reduced global memory access and effective parallelization across threads. At the end of the process, centroids are retrieved from the GPU’s global memory and copied back to the host for final analysis and verification.

## Results/Data Analysis

### Results Summary

| **Dataset Size** | **Sequential CPU (ms)** | **Parallel CPU (ms)** | **Parallel GPU (ms)** | **Speedup (GPU)** |
| ---------------- | ----------------------- | --------------------- | --------------------- | ----------------- |
| 1024             | 22                      | 2                     | 5                     | 4.4x              |
| 32768            | 479                     | 132                   | 8                     | 59.88x            |
| 1048576          | 125163                  | 34260                 | 1325                  | 94.44x            |
| 33554432         | Still computing..       | Still computing..     | 35077                 | infinite          |

Dataset used: 3-dimensional data with 15 clusters.

### Convergence Analysis

Figures depicting centroid trajectories for both implementations showed identical clustering results, confirming correctness.

### Observations

- **Performance Gains**: The GPU version showed significant improvements, particularly for larger datasets, where thread-level parallelism effectively utilized the GPU’s architecture.
- **Challenges**: Inefficient memory access in initial iterations highlighted the importance of proper memory alignment and coalescing. This issue was exacerbated by the use of Arrays of Structures (AoS) instead of Structures of Arrays (SoA), which could have facilitated better memory alignment. However, we employed `struct __align__(4 * sizeof(T)) Point { ... }` to align the structure on 16 or 32 bits, where `T` is either a float or double and `D` is 3 in our example. It is important to note that if `D` changes, this alignment must be redefined.
- **Limitations**: Smaller datasets did not fully utilize GPU resources, leading to less pronounced speedups, and in some cases, they even exhibited worse speedups due to the overhead of GPU memory management and thread synchronization.

## Conclusion

The objective of benchmarking the K-Means algorithm on CPU and GPU implementations was successfully achieved. The GPU-based CUDA implementation demonstrated substantial performance gains, especially with larger datasets, affirming the advantages of parallel computing for iterative machine learning algorithms.

This exercise underscored key lessons:

- Parallelism, when applied judiciously, can significantly reduce computational time.
- CUDA programming demands careful attention to memory management and thread synchronization.

While the GPU implementation excelled in speed, further optimization in memory coalescing and load balancing could unlock additional performance gains. Overall, the project provided valuable insights into high-performance computing and the practical application of CUDA in accelerating machine learning workloads.

## Sources

- https://onlinelibrary.wiley.com/doi/10.1155/2021/9988318
- https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
- https://archive.ics.uci.edu/dataset/1013/synthetic+circle+data+set
- https://leimao.github.io/blog/CUDA-Data-Alignment/