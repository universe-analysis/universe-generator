# Makefile for the standalone CUDA packing engines.
#
# These are plain C++/CUDA programs compiled with nvcc, independent of the
# Rust crate. Requires the CUDA toolkit (nvcc) on PATH.
#
#   make              # build both engines
#   make braid_cuda3d # build just the 3+1 engine
#   make ARCH=sm_89   # override the GPU architecture (default sm_86, Ampere)
#   make clean        # remove the built binaries

NVCC ?= nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -O3 -arch=$(ARCH)

BINARIES := braid_cuda braid_cuda3d

.PHONY: all clean

all: $(BINARIES)

braid_cuda: braid_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

braid_cuda3d: braid_cuda3d.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINARIES) *.o
