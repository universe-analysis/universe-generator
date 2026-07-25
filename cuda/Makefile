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
# -pthread: the 3+1 engine multithreads its host-side work (std::thread).
NVCCFLAGS ?= -O3 -arch=$(ARCH) -Xcompiler -pthread

BINARIES := braid_cuda braid_cuda3d
# Full-spectrum variants (terms = T campaigns): same sources built with a
# larger Path cap. Not part of `all`; the orchestrator builds them on demand.
WIDE := braid_cuda_w150 braid_cuda3d_w150

.PHONY: all wide clean

all: $(BINARIES)

wide: $(WIDE)

braid_cuda: braid_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

braid_cuda3d: braid_cuda3d.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

braid_cuda_w150: braid_cuda.cu
	$(NVCC) $(NVCCFLAGS) -DKMAX_WIGGLE=149 -o $@ $<

braid_cuda3d_w150: braid_cuda3d.cu
	$(NVCC) $(NVCCFLAGS) -DKMAX_WIGGLE=149 -o $@ $<

clean:
	rm -f $(BINARIES) $(WIDE) *.o
