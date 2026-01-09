#include "experiments.cuh"

void experiment1(int numHashes) {
    printf("\n=== Exp1: Insert 2^N into 2^(N+1), %d hash functions ===\n", numHashes);
    printf("%5s %12s %15s\n", "N", "Keys", "M ops/sec");
    
    for (int n = 10; n <= 24; n++) {
        uint32_t numKeys = 1U << n, tableSize = 1U << 25;
        uint32_t maxEvict = (uint32_t)(4 * log2(numKeys));
        bool success;
        double mops = benchmarkInsert(numKeys, tableSize, numHashes, maxEvict, 5, MAX_RESTARTS, &success);
        printf("%5d %12u %15.2f%s\n", n, numKeys, mops, success ? "" : " FAILED");
    }
}

void experiment2(int numHashes) {
    printf("\n=== Exp2: Lookup with varying hit rates, %d hash functions ===\n", numHashes);
    
    uint32_t numKeys = 1U << 24;  // 2^24 keys
    uint32_t tableSize = 1U << 25;
    uint32_t maxEvict = (uint32_t)(4 * log2(numKeys));
    
    printf("Pre-generating valid key set of %u keys...\n", numKeys);
    
    // Allocate host keys array
    uint32_t* h_keys = (uint32_t*)malloc(numKeys * sizeof(uint32_t));
    generateRandomKeys(h_keys, numKeys);
    
    // Iteratively build a valid key set
    int iteration = 0;
    bool allInserted = false;
    
    while (!allInserted) {
        iteration++;
        
        // Upload current keys to GPU
        uint32_t* d_keys = allocAndUploadKeys(h_keys, numKeys);
        
        // Try to insert all keys
        CuckooHashTable ht(tableSize, numHashes, maxEvict);
        bool success = ht.insert(d_keys, numKeys, 1);  // Only 1 attempt, no rehash
        
        if (success) {
            allInserted = true;
            printf("Iteration %d: All %u keys inserted successfully!\n", iteration, numKeys);
            cudaFree(d_keys);
            break;
        }
        
        // Lookup to find which keys were successfully inserted
        int* d_results;
        CUDA_CHECK(cudaMalloc(&d_results, numKeys * sizeof(int)));
        ht.lookup(d_keys, d_results, numKeys);
        
        int* h_results = (int*)malloc(numKeys * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h_results, d_results, numKeys * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Count and collect failed keys, replace them with new keys
        uint32_t failedCount = 0;
        for (uint32_t i = 0; i < numKeys; i++) {
            if (h_results[i] == 0) {  // Key not found = failed to insert
                failedCount++;
                // Generate a new random key to replace the failed one
                uint32_t newKey;
                do {
                    newKey = rand() ^ ((uint32_t)rand() << 16);
                } while (newKey == EMPTY_KEY);
                h_keys[i] = newKey;
            }
        }
        
        printf("Iteration %d: %u keys failed, replacing...\n", iteration, failedCount);
        
        free(h_results);
        cudaFree(d_results);
        cudaFree(d_keys);
        
        if (iteration > 10) {
            printf("Warning: Too many iterations, aborting.\n");
            free(h_keys);
            return;
        }
    }
    
    // Now we have a valid set of keys, run the lookup benchmark
    uint32_t* d_keys = allocAndUploadKeys(h_keys, numKeys);
    CuckooHashTable ht(tableSize, numHashes, maxEvict);
    if (!ht.insert(d_keys, numKeys)) { 
        printf("Final insert failed!\n"); 
        free(h_keys);
        cudaFree(d_keys);
        return; 
    }
    
    printf("\nRunning lookup benchmark with %u keys...\n", numKeys);
    printf("%10s %15s\n", "HitRate%", "M ops/sec");
    for (int hit = 0; hit <= 100; hit += 10)
        printf("%10d %15.2f\n", hit, benchmarkLookup(ht, h_keys, numKeys, numKeys, hit, 5));
    
    free(h_keys); 
    cudaFree(d_keys);
}

void experiment3(int numHashes) {
    printf("\n=== Exp3: Varying load factors, %d hash functions ===\n", numHashes);
    
    uint32_t numKeys = 1U << 20;
    uint32_t maxEvict = (uint32_t)(4 * log2(numKeys));
    double loadFactors[] = {1.01, 1.02, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0};
    int numFactors = sizeof(loadFactors) / sizeof(loadFactors[0]);
    
    printf("%12s %15s\n", "LoadFactor", "M ops/sec");
    for (int i = 0; i < numFactors; i++) {
        uint32_t tableSize = (uint32_t)(numKeys * loadFactors[i]);
        bool success;
        double mops = benchmarkInsert(numKeys, tableSize, numHashes, maxEvict, 5, 50, &success);
        printf("%12.2f %15.2f%s\n", loadFactors[i], mops, success ? "" : " FAILED");
    }
}

void experiment4(int numHashes) {
    printf("\n=== Exp4: Optimal eviction chain length (until first rehash), %d hash functions ===\n", numHashes);
    
    uint32_t numKeys = 1U << 24;  // 2^24 keys
    uint32_t tableSize = (uint32_t)(numKeys * 1.4);  // table size = 1.4n
    uint32_t logN = (uint32_t)log2(numKeys);  // log2(2^24) = 24
    uint32_t multipliers[] = {1, 2, 3, 4, 5, 6, 8, 10};
    int numMults = sizeof(multipliers) / sizeof(multipliers[0]);
    
    printf("%15s %15s %15s\n", "MaxEvict", "M ops/sec", "Status");
    for (int i = 0; i < numMults; i++) {
        uint32_t maxEvict = multipliers[i] * logN;
        bool success;
        // Use maxRestarts=1 so that trial ends as soon as a rehash would be triggered
        double mops = benchmarkInsert(numKeys, tableSize, numHashes, maxEvict, 5, 1, &success);
        printf("%15u %15.2f %15s\n", maxEvict, mops, success ? "SUCCESS" : "REHASH");
    }
}

void runAllExperiments() {
    for (int h = 2; h <= 3; h++) {
        printf("\n====== %d Hash Functions ======\n", h);
        experiment1(h); experiment2(h); experiment3(h); experiment4(h);
    }
}
