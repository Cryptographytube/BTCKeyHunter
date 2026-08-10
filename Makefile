# CGTKEY - Makefile
#
#   make                 build ./cgtkey (fat binary, every arch this nvcc knows)
#   make ARCH=89         build for one arch only (much faster to compile)
#   make selftest        build + run the host self-test
#   make clean
#
# The default build is a fat binary. nvcc compiles the kernel once per arch, so
# the full span takes a few minutes; pass ARCH=<nn> while iterating.

NVCC      ?= nvcc
CXX       ?= g++
O         ?= 3

TARGET    := cgtkey
SELFTEST  := cgtcheck

CU_SRC    := cgtgpu.cu
CXX_SRC   := cgtcli.cpp cgtmath.cpp cgtdigest.cpp cgtpool.cpp cgtpkpool.cpp cgtspan.cpp cgtmulti.cpp
TEST_SRC  := cgtcheck.cpp cgtmath.cpp cgtdigest.cpp cgtpool.cpp cgtpkpool.cpp cgtspan.cpp

# ---------------------------------------------------------------------------
# GPU architectures
#
# Ask the installed nvcc what it supports instead of hardcoding a list: the
# answer differs by toolkit (CUDA 13 dropped Maxwell/Pascal, CUDA 11 has no
# Hopper or Blackwell), and a hardcoded list fails to build on half of them.
# Each arch gets real SASS; the newest also gets PTX so the binary still runs
# on GPUs released after this toolkit.
# ---------------------------------------------------------------------------
ifdef ARCH
  GENCODE := -gencode=arch=compute_$(ARCH),code=sm_$(ARCH)
else
  ARCH_LIST := $(shell $(NVCC) --list-gpu-arch 2>/dev/null | \
                 sed 's/compute_//' | sort -n)
  ifeq ($(ARCH_LIST),)
    # nvcc too old for --list-gpu-arch; fall back to a broad, widely-valid set.
    ARCH_LIST := 50 52 60 61 70 75 80 86
  endif
  NEWEST  := $(lastword $(ARCH_LIST))
  GENCODE := $(foreach a,$(ARCH_LIST),-gencode=arch=compute_$(a),code=sm_$(a)) \
             -gencode=arch=compute_$(NEWEST),code=compute_$(NEWEST)
endif

# --- tuning knobs -----------------------------------------------------------
# Defaults live in cgtdef.h. Override per-build, e.g.
#   make TUNE="-DCGT_STRIDE_HALF=512 -DCGT_ROUNDS=16"
TUNE      ?=

NVCCFLAGS := -O$(O) -std=c++14 $(GENCODE) $(TUNE) \
             -Xptxas -O3 --use_fast_math --extra-device-vectorization
CXXFLAGS  := -O$(O) -std=c++14 $(TUNE)

ifeq ($(OS),Windows_NT)
  BIN      := $(TARGET).exe
  TESTBIN  := $(SELFTEST).exe
  RM       := cmd /c del /q
else
  BIN      := $(TARGET)
  TESTBIN  := $(SELFTEST)
  RM       := rm -f
endif

.PHONY: all selftest clean help

all: $(BIN)

# One nvcc invocation for everything: it drives the host compiler for the .cpp
# files itself, which keeps the host toolchain and flags consistent with the
# device side and avoids link-time ABI mismatches.
$(BIN): $(CU_SRC) $(CXX_SRC) cgtdef.h cgtmath.h cgtdigest.h cgtpool.h cgtpkpool.h cgtspan.h cgtgpu.h cgtwide.h cgtrmd.cuh cgtmulti.h
	$(NVCC) $(NVCCFLAGS) $(CXX_SRC) $(CU_SRC) -o $@

selftest: $(TESTBIN)
	./$(TESTBIN)

$(TESTBIN): $(TEST_SRC) cgtdef.h cgtmath.h cgtdigest.h cgtpool.h cgtpkpool.h cgtspan.h
	$(CXX) $(CXXFLAGS) $(TEST_SRC) -o $@

clean:
	-$(RM) $(BIN) $(TESTBIN) 2>/dev/null || true

help:
	@echo "CGTKEY build targets:"
	@echo "  make                 fat binary for every arch this nvcc supports"
	@echo "  make ARCH=89         single arch (fast rebuild while developing)"
	@echo "  make TUNE=-DCGT_ROUNDS=16"
	@echo "  make selftest        host-side correctness tests"
	@echo "  make clean"
	@echo ""
	@echo "Detected arches: $(ARCH_LIST)"
