#!/bin/bash
# Run all Cuckoo Hashing experiments
set -e
make clean && make
./cuckoo_hash --all 2>&1 | tee "results_$(date +%Y%m%d_%H%M%S).txt"
