# Cuckoo Hashing CUDA - CS121 Lab
NVCC = nvcc
FLAGS = -O3 -arch=sm_60 -std=c++11

SRCDIR = src
BUILDDIR = build
SRCS = $(SRCDIR)/main.cu $(SRCDIR)/common.cu $(SRCDIR)/hashtable.cu \
       $(SRCDIR)/benchmark.cu $(SRCDIR)/experiments.cu $(SRCDIR)/fileio.cu
OBJS = $(patsubst $(SRCDIR)/%.cu,$(BUILDDIR)/%.o,$(SRCS))
TARGET = cuckoo_hash

all: $(BUILDDIR) $(TARGET)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(TARGET): $(OBJS)
	$(NVCC) $(FLAGS) -o $@ $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(FLAGS) -I$(SRCDIR) -c -o $@ $<

clean:
	rm -rf $(TARGET) $(BUILDDIR)

run: $(TARGET)
	./$(TARGET) --all

.PHONY: all clean run
