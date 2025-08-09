# ReservoirAttention vs MultiheadAttention Benchmarks

This repository contains scripts to compare the performance and numerical consistency  
of **PyTorch's `nn.MultiheadAttention` (MHA)** and a custom **`ReservoirAttention`** layer.

The primary script,  
[`benchmark_wikipedia_reservoir_vs_mha_multistep.py`](benchmark_wikipedia_reservoir_vs_mha_multistep.py),  
runs benchmarks on real embeddings from Wikipedia text generated with a Hugging Face  
transformer model (default: `bert-base-uncased`).

---

## Features

- **Full-sequence timing** — one forward pass over the full sequence.
- **Streaming timing**:
  - MHA (prefix recompute)
  - MHA (cached KV)
  - ReservoirAttention (incremental ESN update)
- **Single-step MSE consistency** — verifies streaming vs. full/prefix results match.
- **Multistep closed-loop MSE consistency** — verifies both execution modes stay consistent when rolling out predictions over multiple steps.

---

## Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/) (with CUDA if using GPU)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)
- `networkx` (for ReservoirAttention graph topologies)

Install dependencies:
```bash
pip install torch transformers datasets networkx
```

---

## Running the Benchmark

### CPU
```bash
python benchmarks/benchmark_wikipedia_reservoir_vs_mha_multistep.py --samples 2 --seq_len 256
```
Runs the benchmark on 2 Wikipedia samples, sequence length 256, CPU only.

---

### GPU
```bash
python benchmarks/benchmark_wikipedia_reservoir_vs_mha_multistep.py --device cuda --samples 2 --seq_len 256
```
Runs the benchmark on GPU (if available).

---

### GPU with Mixed Precision
```bash
python benchmarks/benchmark_wikipedia_reservoir_vs_mha_multistep.py \\
  --device cuda --amp --samples 4 --seq_len 512
```
Uses FP16 for embedding/model computation for faster execution and reduced memory use.

---

### Change Reservoir Parameters
```bash
python benchmarks/benchmark_wikipedia_reservoir_vs_mha_multistep.py \\
  --device cuda --amp \\
  --samples 4 --seq_len 512 \\
  --res_size 512 --density 0.005 --topology smallworld
```

---

### Adjust Multistep Rollout Test
```bash
python benchmarks/benchmark_wikipedia_reservoir_vs_mha_multistep.py \\
  --samples 1 --seq_len 256 \\
  --seed_len 32 --horizon 64
```
- **`--seed_len`**: number of initial tokens from the Wikipedia embedding to seed the rollout.
- **`--horizon`**: number of tokens to generate closed-loop by feeding the model's own outputs back in.

---

## Output Example

```
Device=cuda, dtype=torch.float32, samples=2, seq_len=256, embed_model=bert-base-uncased

=== Wikipedia Benchmark ===
Samples: 2 | Seq len: 256 | Embed dim: 768

[Full sequence] (mean per sample)
MHA                : 14.12 ms   | throughput ≈ 18124.7 tok/s
ReservoirAttention : 93.68 ms   | throughput ≈ 2732.7 tok/s

[Streaming] (mean per sample)
MHA (prefix)       : 1788.69 ms
MHA (cached KV)    : 186.81 ms
ReservoirAttention : 103.56 ms

[Single-step MSE consistency on sample 0]
MHA  prefix vs cached : 1.980e-15
RA   full   vs stream : 0.000e+00

[Multistep closed-loop MSE on sample 0]
MHA  prefix vs cached : 4.677e-05  (seed_len=16, horizon=32)
RA   full   vs stream : 3.207e-08  (seed_len=16, horizon=32)
```

