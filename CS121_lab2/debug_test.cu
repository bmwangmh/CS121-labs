#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>

#define EMPTY_KEY 0xFFFFFFFF
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// MurmurHash3 finalizer
__device__ __host__ inline uint32_t murmur3_32(uint32_t key, uint32_t seed) {
    uint32_t h = seed;
    uint32_t k = key;
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    h ^= k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64;
    h ^= 4;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

// xxHash32
__device__ __host__ inline uint32_t xxhash32(uint32_t key, uint32_t seed) {
    const uint32_t PRIME2 = 0x85EBCA77U;
    const uint32_t PRIME3 = 0xC2B2AE3DU;
    const uint32_t PRIME5 = 0x165667B1U;
    uint32_t h = seed + PRIME5 + 4;
    h += key * PRIME3;
    h = ((h << 17) | (h >> 15)) * PRIME2;
    h ^= h >> 15;
    h *= PRIME2;
    h ^= h >> 13;
    h *= PRIME3;
    h ^= h >> 16;
    return h;
}

__device__ inline uint32_t getHash(uint32_t key, uint32_t tableSize, uint32_t hashIdx, uint32_t* seeds) {
    uint32_t h;
    if (hashIdx == 0)
        h = murmur3_32(key, seeds[0]);
    else
        h = xxhash32(key, seeds[1]);
    return h % tableSize;
}

__global__ void clearTableKernel(uint32_t* table, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) table[idx] = EMPTY_KEY;
}

__global__ void insertKernel(uint32_t* table, uint32_t* keys, uint32_t numKeys,
                             uint32_t tableSize, uint32_t numHashes, uint32_t maxEvict,
                             uint32_t* seeds, int* failedCount, uint32_t* failedKeys) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numKeys || keys[idx] == EMPTY_KEY) return;
    
    uint32_t key = keys[idx], hashIdx = 0;
    
    for (uint32_t i = 0; i < maxEvict; i++) {
        uint32_t loc = getHash(key, tableSize, hashIdx, seeds);
        uint32_t prev = atomicExch(&table[loc], key);
        
        if (prev == EMPTY_KEY || prev == key) return;
        
        key = prev;
        // 找到被驱逐key的当前位置对应的hash函数
        for (uint32_t h = 0; h < numHashes; h++) {
            if (getHash(key, tableSize, h, seeds) == loc) {
                hashIdx = (h + 1) % numHashes;
                break;
            }
        }
    }
    
    // 记录失败的key
    int failIdx = atomicAdd(failedCount, 1);
    if (failIdx < 100) {
        failedKeys[failIdx] = keys[idx];  // 记录原始key，不是当前被驱逐的key
    }
}

// 验证表中的内容
__global__ void verifyKernel(uint32_t* table, uint32_t* keys, uint32_t numKeys,
                             uint32_t tableSize, uint32_t numHashes, uint32_t* seeds,
                             int* notFoundCount) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numKeys) return;
    
    uint32_t key = keys[idx];
    for (uint32_t h = 0; h < numHashes; h++) {
        if (table[getHash(key, tableSize, h, seeds)] == key) return;
    }
    atomicAdd(notFoundCount, 1);
}

// 统计表中非空槽位数
__global__ void countFilledKernel(uint32_t* table, uint32_t tableSize, int* count) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tableSize && table[idx] != EMPTY_KEY) {
        atomicAdd(count, 1);
    }
}

// 检测碰撞：两个不同key hash到同一位置
void analyzeHashCollisions(uint32_t* h_keys, uint32_t numKeys, uint32_t tableSize,
                           uint32_t* seeds, uint32_t numHashes) {
    printf("\n=== Hash Collision Analysis ===\n");
    
    // 统计每个hash函数的位置分布
    for (uint32_t h = 0; h < numHashes; h++) {
        uint32_t* counts = (uint32_t*)calloc(tableSize, sizeof(uint32_t));
        int maxCollisions = 0;
        int totalCollisions = 0;
        
        for (uint32_t i = 0; i < numKeys; i++) {
            uint32_t loc;
            if (h == 0) loc = murmur3_32(h_keys[i], seeds[0]) % tableSize;
            else loc = xxhash32(h_keys[i], seeds[1]) % tableSize;
            counts[loc]++;
            if (counts[loc] > 1) totalCollisions++;
            if (counts[loc] > maxCollisions) maxCollisions = counts[loc];
        }
        
        printf("Hash %d: max keys per slot = %d, total collisions = %d\n", 
               h, maxCollisions, totalCollisions);
        free(counts);
    }
    
    // 检查两个hash函数对于同一key是否产生相同位置
    int samePosition = 0;
    for (uint32_t i = 0; i < numKeys; i++) {
        uint32_t loc0 = murmur3_32(h_keys[i], seeds[0]) % tableSize;
        uint32_t loc1 = xxhash32(h_keys[i], seeds[1]) % tableSize;
        if (loc0 == loc1) samePosition++;
    }
    printf("Keys with h0(k) == h1(k): %d (%.4f%%)\n", 
           samePosition, 100.0 * samePosition / numKeys);
}

int main() {
    srand(12345);
    
    uint32_t numKeys = 1U << 20;  // 1M keys
    uint32_t tableSize = numKeys * 100;  // 100x 表大小
    uint32_t numHashes = 2;
    uint32_t maxEvict = (uint32_t)(4 * log2(numKeys));
    
    printf("=== Debug Test ===\n");
    printf("numKeys = %u\n", numKeys);
    printf("tableSize = %u (%.2fN)\n", tableSize, (double)tableSize/numKeys);
    printf("numHashes = %u\n", numHashes);
    printf("maxEvict = %u\n", maxEvict);
    printf("Actual load factor = %.4f%%\n", 100.0 * numKeys / tableSize);
    
    // 生成随机keys
    uint32_t* h_keys = (uint32_t*)malloc(numKeys * sizeof(uint32_t));
    for (uint32_t i = 0; i < numKeys; i++) {
        do { h_keys[i] = rand() ^ (rand() << 16); } while (h_keys[i] == EMPTY_KEY);
    }
    
    // 检查是否有重复key
    printf("\nChecking for duplicate keys...\n");
    // 简单抽样检查
    int duplicates = 0;
    for (uint32_t i = 0; i < 1000; i++) {
        for (uint32_t j = i+1; j < 1000; j++) {
            if (h_keys[i] == h_keys[j]) duplicates++;
        }
    }
    printf("Duplicates in first 1000 keys: %d\n", duplicates);
    
    uint32_t seeds[3] = {0x12345678, 0x87654321, 0xDEADBEEF};
    
    // 分析hash碰撞
    analyzeHashCollisions(h_keys, numKeys, tableSize, seeds, numHashes);
    
    // GPU内存分配
    uint32_t *d_table, *d_keys, *d_seeds, *d_failedKeys;
    int *d_failedCount, *d_notFoundCount, *d_filledCount;
    
    CUDA_CHECK(cudaMalloc(&d_table, tableSize * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_keys, numKeys * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_seeds, 3 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_failedCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_failedKeys, 100 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_notFoundCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_filledCount, sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, numKeys * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seeds, seeds, 3 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    // 清空表
    clearTableKernel<<<(tableSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_table, tableSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 插入
    CUDA_CHECK(cudaMemset(d_failedCount, 0, sizeof(int)));
    int blocks = (numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("\n=== Inserting... ===\n");
    insertKernel<<<blocks, BLOCK_SIZE>>>(d_table, d_keys, numKeys, tableSize, 
                                          numHashes, maxEvict, d_seeds, d_failedCount, d_failedKeys);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int failedCount;
    CUDA_CHECK(cudaMemcpy(&failedCount, d_failedCount, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Failed insertions: %d\n", failedCount);
    
    if (failedCount > 0) {
        uint32_t failedKeys[100];
        CUDA_CHECK(cudaMemcpy(failedKeys, d_failedKeys, 100 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        printf("First few failed keys: ");
        for (int i = 0; i < (failedCount < 10 ? failedCount : 10); i++) {
            printf("%u ", failedKeys[i]);
        }
        printf("\n");
        
        // 分析失败key的hash位置
        printf("\nAnalyzing failed keys:\n");
        for (int i = 0; i < (failedCount < 5 ? failedCount : 5); i++) {
            uint32_t k = failedKeys[i];
            uint32_t loc0 = murmur3_32(k, seeds[0]) % tableSize;
            uint32_t loc1 = xxhash32(k, seeds[1]) % tableSize;
            printf("Key %u: h0=%u, h1=%u, same=%d\n", k, loc0, loc1, loc0==loc1);
        }
    }
    
    // 验证所有key是否在表中
    CUDA_CHECK(cudaMemset(d_notFoundCount, 0, sizeof(int)));
    verifyKernel<<<blocks, BLOCK_SIZE>>>(d_table, d_keys, numKeys, tableSize, 
                                          numHashes, d_seeds, d_notFoundCount);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int notFoundCount;
    CUDA_CHECK(cudaMemcpy(&notFoundCount, d_notFoundCount, sizeof(int), cudaMemcpyDeviceToHost));
    printf("\n=== Verification ===\n");
    printf("Keys not found in table: %d\n", notFoundCount);
    
    // 统计表中填充的槽位
    CUDA_CHECK(cudaMemset(d_filledCount, 0, sizeof(int)));
    countFilledKernel<<<(tableSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_table, tableSize, d_filledCount);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int filledCount;
    CUDA_CHECK(cudaMemcpy(&filledCount, d_filledCount, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Filled slots in table: %d (expected: %u)\n", filledCount, numKeys);
    printf("Difference: %d\n", (int)numKeys - filledCount);
    
    // 清理
    cudaFree(d_table);
    cudaFree(d_keys);
    cudaFree(d_seeds);
    cudaFree(d_failedCount);
    cudaFree(d_failedKeys);
    cudaFree(d_notFoundCount);
    cudaFree(d_filledCount);
    free(h_keys);
    
    return 0;
}
