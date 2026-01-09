#ifndef COMMON_CUH
#define COMMON_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define EMPTY_KEY 0xFFFFFFFF
#define MAX_RESTARTS 100
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Utility function declarations
double getTimeMs();
void generateRandomKeys(uint32_t* keys, uint32_t n);
uint32_t* allocAndUploadKeys(uint32_t* h_keys, uint32_t n);
void printGPUInfo();

#endif // COMMON_CUH
