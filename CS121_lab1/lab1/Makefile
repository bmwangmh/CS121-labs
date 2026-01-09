CXX = clang++
CXXFLAGS = -g -O2 -fopenmp -march=native -fsanitize=address -std=c++17

BUILD_DIR = ./build
BIN = $(BUILD_DIR)/test

BASELINE_BIN = $(BUILD_DIR)/baseline

$(shell mkdir -p $(BUILD_DIR))

$(BIN): src/main.cpp src/prefix.cpp
	@$(CXX) $(CXXFLAGS) $^ -o $@

$(BASELINE_BIN): src/baseline.cpp
	@$(CXX) $(CXXFLAGS) $^ -o $@

run: $(BIN)	
	@$^

run-baseline: $(BASELINE_BIN)
	@$(BASELINE_BIN)

clean: 
	-@rm -rf $(BUILD_DIR)

.PHONY: run clean