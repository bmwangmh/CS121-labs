#include "common.cuh"

double getTimeMs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

void generateRandomKeys(uint32_t* keys, uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        do { keys[i] = rand() ^ (rand() << 16); } while (keys[i] == EMPTY_KEY);
}

uint32_t* allocAndUploadKeys(uint32_t* h_keys, uint32_t n) {
    uint32_t* d_keys;
    CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    return d_keys;
}

void printGPUInfo() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s, SM %d.%d, %.2f GB\n\n", prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e9);
}
