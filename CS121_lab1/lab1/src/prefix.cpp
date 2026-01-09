#include <vector>
#include <omp.h>

std::vector<int> prefix_sum(const int* arr, int n) {
    std::vector<int> prefix;
    if (n <= 0) return prefix;

    prefix.assign((size_t)n, 0);

    int max_threads = omp_get_max_threads();
    std::vector<int> chunk_sums((size_t)max_threads, 0);
    std::vector<int> offsets((size_t)max_threads, 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = (n + num_threads - 1) / num_threads;
        int start = tid * chunk_size;
        int end = (start + chunk_size < n) ? start + chunk_size : n;

        if (start < n) {
            prefix[start] = arr[start];
            for (int i = start + 1; i < end; ++i) {
                prefix[i] = prefix[i - 1] + arr[i];
            }
            chunk_sums[tid] = prefix[end - 1];
        } else {
            chunk_sums[tid] = 0;
        }

        #pragma omp barrier

        #pragma omp single
        {
            offsets[0] = 0;
            for (int t = 1; t < num_threads; ++t) {
                offsets[t] = offsets[t - 1] + chunk_sums[t - 1];
            }
        }

        #pragma omp barrier

        int add = offsets[tid];
        for (int i = start; i < end; ++i) {
            prefix[i] += add;
        }
        
    }

    return prefix;
}