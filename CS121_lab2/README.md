# CS121 Cuckoo Hashing - CUDA Implementation

## Project Structure
```
├── src/
│   ├── main.cu         # Entry point
│   ├── common.cuh/cu   # Constants, macros, utilities
│   ├── hashtable.cuh/cu# CuckooHashTable class & kernels
│   ├── benchmark.cuh/cu# Benchmarking functions
│   ├── experiments.cuh/cu # Experiment implementations
│   └── fileio.cuh/cu   # File I/O operations
├── Makefile
└── README.md
```

## Build & Run
```bash
make                              # Compile
./cuckoo_hash --all               # Run all experiments
./cuckoo_hash --exp1 --hash2      # Single experiment
./cuckoo_hash --file in.txt q.txt out.txt  # File mode
```

## Algorithm
- **Hash**: MurmurHash3 with different seeds per function
- **Insert**: Parallel with atomic exchange, auto-rehash on failure  
- **Lookup**: Check all hash locations in parallel

Adjust `-arch=sm_60` in Makefile for your GPU.
