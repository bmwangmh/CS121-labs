#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

#include "hashtable.cuh"

// Benchmark functions
double benchmarkInsert(uint32_t numKeys, uint32_t tableSize, uint32_t numHashes,
                       uint32_t maxEvict, int runs, int maxRestarts, bool* success);

double benchmarkLookup(CuckooHashTable& ht, uint32_t* h_keys, uint32_t numKeys,
                       uint32_t numQueries, int hitPercent, int runs);

#endif // BENCHMARK_CUH
