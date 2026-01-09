#ifndef HASHTABLE_CUH
#define HASHTABLE_CUH

#include "common.cuh"

// ============================================================================
// Hash Functions - 使用不同的hash算法以减少相关性
// ============================================================================

// MurmurHash3 finalizer (完整版)
__device__ __host__ inline uint32_t murmur3_32(uint32_t key, uint32_t seed) {
    uint32_t h = seed;
    uint32_t k = key;
    
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    
    h ^= k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64;
    
    // finalization
    h ^= 4;  // length
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    
    return h;
}

// xxHash32 风格的hash
__device__ __host__ inline uint32_t xxhash32(uint32_t key, uint32_t seed) {
    const uint32_t PRIME1 = 0x9E3779B1U;
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

// FNV-1a hash
__device__ __host__ inline uint32_t fnv1a(uint32_t key, uint32_t seed) {
    uint32_t h = 2166136261U ^ seed;
    uint8_t* bytes = (uint8_t*)&key;
    for (int i = 0; i < 4; i++) {
        h ^= bytes[i];
        h *= 16777619U;
    }
    return h;
}

// 根据hashIdx选择不同的hash函数
__device__ inline uint32_t getHash(uint32_t key, uint32_t tableSize, uint32_t hashIdx,
                                   uint32_t numHashes, uint32_t* seeds) {
    uint32_t h;
    switch (hashIdx % 3) {
        case 0:  h = murmur3_32(key, seeds[0]); break;
        case 1:  h = xxhash32(key, seeds[1]); break;
        default: h = fnv1a(key, seeds[2]); break;
    }
    return h % tableSize;
}

// Kernel declarations
__global__ void clearTableKernel(uint32_t* table, uint32_t size);
__global__ void insertKernel(uint32_t* table, uint32_t* keys, uint32_t numKeys,
                             uint32_t tableSize, uint32_t numHashes, uint32_t maxEvict,
                             uint32_t* seeds, int* failed);
__global__ void lookupKernel(uint32_t* table, uint32_t* queries, int* results,
                             uint32_t numQueries, uint32_t tableSize,
                             uint32_t numHashes, uint32_t* seeds);

// Hash Table Class
class CuckooHashTable {
    uint32_t *d_table, *d_seeds;
    uint32_t tableSize, numHashes, maxEvict;
    uint32_t seeds[3];
    
    void uploadSeeds();
    
public:
    CuckooHashTable(uint32_t size, uint32_t hashes, uint32_t evict);
    ~CuckooHashTable();
    
    void clear();
    void rehash();
    bool insert(uint32_t* d_keys, uint32_t numKeys, int maxRestarts = MAX_RESTARTS);
    void lookup(uint32_t* d_queries, int* d_results, uint32_t numQueries);
};

#endif // HASHTABLE_CUH
