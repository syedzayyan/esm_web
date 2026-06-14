# Builds against a ggml installation (e.g. `brew install ggml`).
GGML_PREFIX ?= /opt/homebrew

CXX      = c++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -I$(GGML_PREFIX)/include
LDFLAGS  = -L$(GGML_PREFIX)/lib -lggml -lggml-cpu -lggml-base -Wl,-rpath,$(GGML_PREFIX)/lib

TARGET = esm2
SRCS   = main.cpp esm2.cpp

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRCS) esm2.h tokenizer.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)
