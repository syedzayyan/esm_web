# Builds against a ggml installation (e.g. `brew install ggml`).
GGML_PREFIX ?= /opt/homebrew

CXX      = c++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -I$(GGML_PREFIX)/include

# On macOS, link the Metal backend automatically; Linux/other uses CPU only.
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  LDFLAGS = -L$(GGML_PREFIX)/lib -lggml -lggml-cpu -lggml-metal -lggml-base \
            -framework Metal -framework Foundation \
            -Wl,-rpath,$(GGML_PREFIX)/lib
else
  LDFLAGS = -L$(GGML_PREFIX)/lib -lggml -lggml-cpu -lggml-base -Wl,-rpath,$(GGML_PREFIX)/lib
endif

ESM2_TARGET   = esm2
ESM2_SRCS     = main.cpp esm2.cpp

BENCH_TARGET  = bench
BENCH_SRCS    = bench.cpp esm2.cpp

TRUNK_TARGET  = test_trunk
TRUNK_SRCS    = test_trunk.cpp esmfold_trunk.cpp

.PHONY: all clean
all: $(ESM2_TARGET) $(BENCH_TARGET) $(TRUNK_TARGET)

$(ESM2_TARGET): $(ESM2_SRCS) esm2.h tokenizer.h ggml_common.h
	$(CXX) $(CXXFLAGS) -o $(ESM2_TARGET) $(ESM2_SRCS) $(LDFLAGS)

$(BENCH_TARGET): $(BENCH_SRCS) esm2.h tokenizer.h ggml_common.h
	$(CXX) $(CXXFLAGS) -o $(BENCH_TARGET) $(BENCH_SRCS) $(LDFLAGS)

$(TRUNK_TARGET): $(TRUNK_SRCS) esmfold_trunk.h ggml_common.h
	$(CXX) $(CXXFLAGS) -o $(TRUNK_TARGET) $(TRUNK_SRCS) $(LDFLAGS)

clean:
	rm -f $(ESM2_TARGET) $(BENCH_TARGET) $(TRUNK_TARGET)
