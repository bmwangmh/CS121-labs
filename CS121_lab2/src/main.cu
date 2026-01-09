/**
 * CS121 Parallel Computing CUDA Lab - Cuckoo Hashing
 * Main entry point
 */

#include "experiments.cuh"
#include "fileio.cuh"

int main(int argc, char** argv) {
    srand(time(NULL));
    printGPUInfo();
    
    if (argc < 2) {
        printf("Usage: %s [--all|--exp1|--exp2|--exp3|--exp4] [--hash2|--hash3]\n", argv[0]);
        printf("       %s --file <insert> <query> <output> [--hash2|--hash3]\n", argv[0]);
        return 0;
    }
    
    int numHashes = 2;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--hash3") == 0) numHashes = 3;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--all") == 0) runAllExperiments();
        else if (strcmp(argv[i], "--exp1") == 0) experiment1(numHashes);
        else if (strcmp(argv[i], "--exp2") == 0) experiment2(numHashes);
        else if (strcmp(argv[i], "--exp3") == 0) experiment3(numHashes);
        else if (strcmp(argv[i], "--exp4") == 0) experiment4(numHashes);
        else if (strcmp(argv[i], "--file") == 0 && i + 3 < argc) {
            fileOperations(argv[i+1], argv[i+2], argv[i+3], numHashes);
            i += 3;
        }
    }
    
    return 0;
}
