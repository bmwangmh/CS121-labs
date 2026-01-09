#include "fileio.cuh"

uint32_t* loadKeys(const char* file, uint32_t* n) {
    FILE* f = fopen(file, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", file); return NULL; }
    
    uint32_t cap = 1024, size = 0;
    uint32_t* keys = (uint32_t*)malloc(cap * sizeof(uint32_t));
    
    while (fscanf(f, "%u", &keys[size]) == 1) {
        if (++size >= cap) keys = (uint32_t*)realloc(keys, (cap *= 2) * sizeof(uint32_t));
    }
    fclose(f);
    *n = size;
    return keys;
}

void fileOperations(const char* insertFile, const char* queryFile, 
                    const char* outFile, int numHashes) {
    uint32_t numInsert, numQuery;
    uint32_t* h_insert = loadKeys(insertFile, &numInsert);
    uint32_t* h_query = loadKeys(queryFile, &numQuery);
    if (!h_insert || !h_query) return;
    
    printf("Loaded %u insert keys, %u query keys\n", numInsert, numQuery);
    
    uint32_t* d_insert = allocAndUploadKeys(h_insert, numInsert);
    CuckooHashTable ht(numInsert * 2, numHashes, (uint32_t)(4 * log2(numInsert)));
    
    double start = getTimeMs();
    bool ok = ht.insert(d_insert, numInsert);
    printf("Insert: %.2f ms %s\n", getTimeMs() - start, ok ? "" : "(FAILED)");
    
    uint32_t* d_query = allocAndUploadKeys(h_query, numQuery);
    int* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, numQuery * sizeof(int)));
    
    start = getTimeMs();
    ht.lookup(d_query, d_results, numQuery);
    printf("Lookup: %.2f ms\n", getTimeMs() - start);
    
    int* h_results = (int*)malloc(numQuery * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_results, d_results, numQuery * sizeof(int), cudaMemcpyDeviceToHost));
    
    FILE* f = fopen(outFile, "w");
    uint32_t found = 0;
    for (uint32_t i = 0; i < numQuery; i++) { 
        fprintf(f, "%d\n", h_results[i]); 
        found += h_results[i]; 
    }
    fclose(f);
    
    printf("Found: %u/%u (%.1f%%)\n", found, numQuery, 100.0 * found / numQuery);
    
    free(h_insert); free(h_query); free(h_results);
    cudaFree(d_insert); cudaFree(d_query); cudaFree(d_results);
}
