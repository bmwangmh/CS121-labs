#include "hashtable.cuh"

__device__ inline uint32_t deviceRand(uint32_t& state) {
    state = state * 1103515245 + 12345;
    return (state >> 16) & 0x7fff;
}

// CUDA Kernels
__global__ void clearTableKernel(uint32_t* table, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) table[idx] = EMPTY_KEY;
}

__global__ void insertKernel(uint32_t* table, uint32_t* keys, uint32_t numKeys,
                             uint32_t tableSize, uint32_t numHashes, uint32_t maxEvict,
                             uint32_t* seeds, int* failed) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numKeys || keys[idx] == EMPTY_KEY) return;
    
    uint32_t key = keys[idx];
    
    // 随机数状态，用于打破对称性
    uint32_t randState = idx * 1099511628211ULL + clock64();
    
    // 先检查key是否已存在
    for (uint32_t h = 0; h < numHashes; h++) {
        uint32_t loc = getHash(key, tableSize, h, numHashes, seeds);
        if (table[loc] == key) return;
    }

    uint32_t curHashIdx = idx % numHashes;  // 不同线程从不同hash开始，减少冲突
    
    for (uint32_t i = 0; i < maxEvict; i++) {
        uint32_t loc = getHash(key, tableSize, curHashIdx, numHashes, seeds);
        
        // 优化：先用普通读检查是否为空，避免不必要的atomic操作
        uint32_t existing = table[loc];
        if (existing == EMPTY_KEY) {
            // 位置为空，尝试CAS插入
            uint32_t prev = atomicCAS(&table[loc], EMPTY_KEY, key);
            if (prev == EMPTY_KEY || prev == key) return;
            // CAS失败说明被其他线程抢占，继续驱逐逻辑
            existing = prev;
        }
        
        if (existing == key) return;  // key已存在
        
        // 位置被占用，执行驱逐
        uint32_t prev = atomicExch(&table[loc], key);
        
        if (prev == EMPTY_KEY || prev == key) return;
        
        // 被驱逐，需要重新安置
        key = prev;
        
        // 找到被驱逐key对应的hash索引
        bool foundSrc = false;
        for (uint32_t h = 0; h < numHashes; h++) {
            if (getHash(key, tableSize, h, numHashes, seeds) == loc) {
                curHashIdx = (h + 1) % numHashes;
                foundSrc = true;
                break;
            }
        }
        
        if (!foundSrc) {
            // 并发导致的异常情况，随机选择
            curHashIdx = deviceRand(randState) % numHashes;
        }
    }
    
    atomicExch(failed, 1);
}

__global__ void lookupKernel(uint32_t* table, uint32_t* queries, int* results,
                             uint32_t numQueries, uint32_t tableSize,
                             uint32_t numHashes, uint32_t* seeds) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numQueries) return;
    
    uint32_t key = queries[idx];
    results[idx] = 0;
    
    for (uint32_t h = 0; h < numHashes; h++) {
        if (table[getHash(key, tableSize, h, numHashes, seeds)] == key) {
            results[idx] = 1;
            return;
        }
    }
}

// CuckooHashTable implementation
void CuckooHashTable::uploadSeeds() {
    CUDA_CHECK(cudaMemcpy(d_seeds, seeds, 3 * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

CuckooHashTable::CuckooHashTable(uint32_t size, uint32_t hashes, uint32_t evict) 
    : tableSize(size), numHashes(hashes), maxEvict(evict) {
    seeds[0] = 0x12345678; seeds[1] = 0x87654321; seeds[2] = 0xDEADBEEF;
    CUDA_CHECK(cudaMalloc(&d_table, tableSize * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_seeds, 3 * sizeof(uint32_t)));
    uploadSeeds();
    clear();
}

CuckooHashTable::~CuckooHashTable() { 
    cudaFree(d_table); 
    cudaFree(d_seeds); 
}

void CuckooHashTable::clear() {
    clearTableKernel<<<(tableSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_table, tableSize);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CuckooHashTable::rehash() {
    for (int i = 0; i < 3; i++) seeds[i] = rand();
    uploadSeeds();
}

bool CuckooHashTable::insert(uint32_t* d_keys, uint32_t numKeys, int maxRestarts) {
    int *d_failed, failed;
    CUDA_CHECK(cudaMalloc(&d_failed, sizeof(int)));
    int blocks = (numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int r = 0; r < maxRestarts; r++) {
        clear();
        CUDA_CHECK(cudaMemset(d_failed, 0, sizeof(int)));
        insertKernel<<<blocks, BLOCK_SIZE>>>(d_table, d_keys, numKeys, tableSize, 
                                              numHashes, maxEvict, d_seeds, d_failed);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (!failed) { cudaFree(d_failed); return true; }
        rehash();
    }
    cudaFree(d_failed);
    return false;
}

void CuckooHashTable::lookup(uint32_t* d_queries, int* d_results, uint32_t numQueries) {
    lookupKernel<<<(numQueries + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
        (d_table, d_queries, d_results, numQueries, tableSize, numHashes, d_seeds);
    CUDA_CHECK(cudaDeviceSynchronize());
}
