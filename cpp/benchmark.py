#!/usr/bin/env python3
"""Benchmark HuggingFace ESM2 on CPU — prints the same table as cpp/bench.

Usage:
    python3 cpp/benchmark.py [--model facebook/esm2_t6_8M_UR50D]
    python3 cpp/benchmark.py --warmup 3 --runs 10

Compare directly against the C/ggml binary:
    ./cpp/bench chp/esm/esm2-f32.gguf
"""
import argparse, statistics, time
import torch
from transformers import EsmModel, EsmTokenizer

LENGTHS = [10, 50, 100, 200, 500]

def make_seq(n):
    aa = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(aa[i % 20] for i in range(n))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs",   type=int, default=10)
    ap.add_argument("--device", default="auto",
                    help="cpu | mps | cuda | auto  (auto picks MPS on Apple Silicon)")
    args = ap.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    t0 = time.perf_counter()
    tokenizer = EsmTokenizer.from_pretrained(args.model)
    model     = EsmModel.from_pretrained(args.model)
    model.eval()
    model.to(device)
    print(f"device     : {device}")
    print(f"model load : {(time.perf_counter()-t0)*1e3:.1f} ms\n")

    print(f"{'len':<6}  {'mean_ms':>8}  {'min_ms':>8}  {'max_ms':>8}  {'res/sec':>12}"
          f"  (warmup={args.warmup}, runs={args.runs})")
    print(f"{'------':<6}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'------------':>12}")

    for L in LENGTHS:
        seq    = make_seq(L)
        inputs = tokenizer(seq, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            for _ in range(args.warmup):
                model(**inputs)

        # Synchronize GPU before/after each timed call so we measure actual
        # wall-clock latency rather than async dispatch time.
        def sync():
            if device.type == "mps":   torch.mps.synchronize()
            elif device.type == "cuda": torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(args.runs):
                sync()
                t = time.perf_counter()
                model(**inputs)
                sync()
                times.append((time.perf_counter() - t) * 1e3)

        mean_ms = statistics.mean(times)
        min_ms  = min(times)
        max_ms  = max(times)
        print(f"{L:<6}  {mean_ms:>8.1f}  {min_ms:>8.1f}  {max_ms:>8.1f}  {L/(mean_ms/1e3):>12.0f}")

if __name__ == "__main__":
    main()
