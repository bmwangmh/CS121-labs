#ifndef FILEIO_CUH
#define FILEIO_CUH

#include "hashtable.cuh"

uint32_t* loadKeys(const char* file, uint32_t* n);
void fileOperations(const char* insertFile, const char* queryFile, 
                    const char* outFile, int numHashes);

#endif // FILEIO_CUH
