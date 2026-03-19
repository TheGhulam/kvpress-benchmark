# kvpress Benchmark

Companion code for the blog post **"Benchmarking KV Cache Compression Methods in kvpress"**.

Benchmarks seven KV-cache compression methods from the [kvpress](https://github.com/NVIDIA/kvpress) library across accuracy, peak GPU memory, and prefill latency — using a needle-in-a-haystack evaluation at multiple compression ratios and context lengths.

A second script stacks KV-cache quantization (int4 / int2) on top of token pruning to explore additive memory savings and accuracy trade-offs.

---

## Results

Pre-computed results are in `results/` (CSV + plots). Generated on an RTX PRO 5000 Blackwell (48 GB VRAM) with `torch==2.12.0.dev20260311+cu128`.

| Script | Results |
|--------|---------|
| `benchmark.py` | `results/benchmark_results_*.csv` |
| `benchmark_quant_stack.py` | `results/quant_stack/quant_stack_results.csv` |

---

## Methods benchmarked

| Method | Notes |
|--------|-------|
| `ExpectedAttentionPress` | Attention-score based pruning |
| `AdaKVPress` | Adaptive per-head budgets |
| `SnapKVPress` | Observation window pruning |
| `KnormPress` | Key-norm based pruning |
| `CriticalKVPress` | Critical token selection |
| `ThinKPress` | Key-channel compression |
| `StreamingLLMPress` | Sink-token streaming |

---

## Setup

```bash
conda env create -f environment.yml
conda activate kvpress-bench
```

> **Note:** The environment pins `torch==2.12.0.dev20260311+cu128` (CUDA 12.8 nightly) to reproduce the exact published results. Adjust `environment.yml` for your CUDA version.

You will also need a HuggingFace account with access to `meta-llama/Llama-3.1-8B-Instruct`.

---

## Usage

### 1. Download model and datasets

```bash
python scripts/download_data.py
```

### 2. Run the main benchmark

```bash
python scripts/benchmark.py
# or with a custom config:
python scripts/benchmark.py --config configs/benchmark_config.yaml
```

### 3. Run the quantization stacking benchmark

```bash
python scripts/benchmark_quant_stack.py --config configs/quant_stack.yaml
```

### 4. Profile a single press

```bash
python scripts/profile_memory.py --press SnapKVPress --ratio 0.3 --context-length 16384
```

---

## Configs

| File | Description |
|------|-------------|
| `configs/benchmark_config.yaml` | Full sweep — 7 presses × 9 ratios, 5 runs each |
| `configs/benchmark_config_quick.yaml` | Fast sweep — 4 ratios, 3 runs, 8K context |
| `configs/quant_stack.yaml` | Quantization × pruning stacking experiment |

---

## Project structure

```
kvpress-benchmark/
├── scripts/
│   ├── benchmark.py               # Main benchmark loop
│   ├── benchmark_quant_stack.py   # Quant × pruning stacking
│   ├── download_data.py           # Download model + datasets
│   └── profile_memory.py          # Single-press memory profiler
├── configs/
│   ├── benchmark_config.yaml      # Full benchmark config
│   ├── benchmark_config_quick.yaml# Quick sweep config
│   └── quant_stack.yaml           # Quant stacking config
├── results/                       # Pre-computed CSVs and plots
├── images/                        # Figures used in the blog post
└── environment.yml
```

---

## License

MIT
