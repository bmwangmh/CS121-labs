#include "benchmark.cuh"

double benchmarkInsert(uint32_t numKeys, uint32_t tableSize, uint32_t numHashes,
                       uint32_t maxEvict, int runs, int maxRestarts, bool* success) {
    double totalTime = 0;
    int successRuns = 0;
    
    for (int r = 0; r < runs; r++) {
        uint32_t* h_keys = (uint32_t*)malloc(numKeys * sizeof(uint32_t));
        generateRandomKeys(h_keys, numKeys);
        uint32_t* d_keys = allocAndUploadKeys(h_keys, numKeys);
        
        CuckooHashTable ht(tableSize, numHashes, maxEvict);
        cudaDeviceSynchronize();
        
        double start = getTimeMs();
        bool ok = ht.insert(d_keys, numKeys, maxRestarts);
        double elapsed = getTimeMs() - start;
        
        if (ok && elapsed < 30000) { totalTime += elapsed; successRuns++; }
        
        free(h_keys); cudaFree(d_keys);
    }
    
    *success = successRuns > 0;
    return successRuns ? (numKeys / 1e6) / (totalTime / successRuns / 1000.0) : 0;
}

double benchmarkLookup(CuckooHashTable& ht, uint32_t* h_keys, uint32_t numKeys,
                       uint32_t numQueries, int hitPercent, int runs) {
    double totalTime = 0;
    
    for (int r = 0; r < runs; r++) {
        uint32_t* h_queries = (uint32_t*)malloc(numQueries * sizeof(uint32_t));
        uint32_t fromTable = numQueries * hitPercent / 100;
        
        for (uint32_t i = 0; i < fromTable; i++) h_queries[i] = h_keys[rand() % numKeys];
        for (uint32_t i = fromTable; i < numQueries; i++)
            do { h_queries[i] = rand() ^ (rand() << 16); } while (h_queries[i] == EMPTY_KEY);
        
        // Shuffle
        for (uint32_t i = numQueries - 1; i > 0; i--) {
            uint32_t j = rand() % (i + 1);
            uint32_t t = h_queries[i]; h_queries[i] = h_queries[j]; h_queries[j] = t;
        }
        
        uint32_t* d_queries = allocAndUploadKeys(h_queries, numQueries);
        int* d_results;
        CUDA_CHECK(cudaMalloc(&d_results, numQueries * sizeof(int)));
        
        cudaDeviceSynchronize();
        double start = getTimeMs();
        ht.lookup(d_queries, d_results, numQueries);
        totalTime += getTimeMs() - start;
        
        free(h_queries); cudaFree(d_queries); cudaFree(d_results);
    }
    
    return (numQueries / 1e6) / (totalTime / runs / 1000.0);
}
