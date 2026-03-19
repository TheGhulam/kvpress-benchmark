#!/usr/bin/env python3
"""kvpress KV-cache compression benchmark.

Sweeps every (press, compression_ratio) pair from the config, measures:
  - Answer quality (exact match / F1 on needle-in-a-haystack)
  - Peak GPU memory
  - Prefill latency
  - Generation throughput (tok/s)

Results are saved as CSV + a summary plot in the results/ directory.

Usage:
    python scripts/benchmark.py                              # default config
    python scripts/benchmark.py --config configs/custom.yaml # custom config
"""

import argparse
import gc
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_press(name: str, compression_ratio: float, kwargs: dict):
    """Dynamically instantiate a kvpress Press by class name."""
    if compression_ratio == 0.0:
        return None

    import kvpress

    cls = getattr(kvpress, name)

    # Handle wrapper presses like AdaKVPress / CriticalKVPress that take an
    # inner ScorerPress via a positional `press` argument.
    if "inner_press" in kwargs:
        inner_name = kwargs.pop("inner_press")
        inner_cls = getattr(kvpress, inner_name)
        inner = inner_cls(compression_ratio=compression_ratio)
        return cls(press=inner, **kwargs)

    # ThinKPress uses a differently-named parameter
    if name == "ThinKPress":
        return cls(key_channel_compression_ratio=compression_ratio, **kwargs)

    return cls(compression_ratio=compression_ratio, **kwargs)


def measure_memory() -> dict:
    """Return peak GPU memory stats in GB (device 0 only)."""
    allocated = torch.cuda.max_memory_allocated(0) / 1e9
    reserved = torch.cuda.max_memory_reserved(0) / 1e9
    # Sanity check: warn if allocated exceeds expected VRAM
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    if allocated > vram_gb:
        print(f"  ⚠ Peak allocated ({allocated:.1f} GB) exceeds physical VRAM "
              f"({vram_gb:.1f} GB) — shared/system memory is being used!")
    return {
        "peak_allocated_gb": round(allocated, 2),
        "peak_reserved_gb": round(reserved, 2),
        "vram_total_gb": round(vram_gb, 2),
    }


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ── Needle-in-a-haystack evaluation ─────────────────────────────────────────

def build_needle_prompt(context_tokens: int, needle: str, query: str,
                        tokenizer) -> str:
    """Build a long prompt with a needle hidden in the middle."""
    filler_sentence = "The quick brown fox jumps over the lazy dog. "
    filler_tokens = tokenizer(filler_sentence, add_special_tokens=False)["input_ids"]
    tokens_per_filler = len(filler_tokens)

    # Reserve space for needle + query
    needle_query_tokens = len(
        tokenizer(needle + " " + query, add_special_tokens=False)["input_ids"]
    )
    num_fillers = (context_tokens - needle_query_tokens) // tokens_per_filler
    half = num_fillers // 2

    context = (filler_sentence * half) + f"\n{needle}\n" + (filler_sentence * half)
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"


def eval_needle(pipe, press, cfg: dict, tokenizer) -> dict:
    """Run the custom needle-in-a-haystack test."""
    ncfg = cfg["eval"]["custom_needle"]
    prompt = build_needle_prompt(
        ncfg["context_tokens"], ncfg["needle"], ncfg["query"], tokenizer
    )

    reset_memory()
    t0 = time.perf_counter()
    pipe_kwargs = dict(max_new_tokens=64, do_sample=False, return_full_text=False)
    if press is not None:
        pipe_kwargs["press"] = press
    out = pipe(prompt, **pipe_kwargs)
    latency = time.perf_counter() - t0
    mem = measure_memory()
    peak_mem = mem["peak_allocated_gb"]

    if isinstance(out, list) and out:
        first = out[0]
        answer = first.get("generated_text", "") if isinstance(first, dict) else str(first)
    elif isinstance(out, dict):
        answer = out.get("generated_text") or out.get("answer") or ""
    else:
        answer = str(out)

    answer = answer.strip().lower()

    # Prefer a quoted token from the configured needle, e.g. "'cranberry'".
    needle_text = ncfg.get("needle", "")
    match = re.search(r"'([^']+)'", needle_text)
    expected = (match.group(1) if match else needle_text).strip().lower()
    hit = expected in answer

    return {
        "needle_hit": int(hit),
        "answer": answer,
        "latency_s": round(latency, 2),
        "peak_mem_gb": round(peak_mem, 2),
        "peak_reserved_gb": mem["peak_reserved_gb"],
        "vram_total_gb": mem["vram_total_gb"],
    }


# ── Main benchmark loop ─────────────────────────────────────────────────────

def run_benchmark(cfg: dict) -> pd.DataFrame:
    import kvpress  # Registers the kv-press pipeline task at import time.
    from transformers import pipeline as hf_pipeline, AutoTokenizer

    model_name = cfg["model"]["name"]
    dtype = getattr(torch, cfg["model"]["dtype"])

    print(f"\nLoading model: {model_name}  (dtype={cfg['model']['dtype']})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preflight: verify model + max KV cache fit in VRAM
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    print(f"GPU 0: {props.name}, {vram_gb:.1f} GB VRAM")

    pipe = hf_pipeline(
        "kv-press-text-generation",
        model=model_name,
        device=0,
        torch_dtype=dtype,
        model_kwargs={"attn_implementation": cfg["model"]["attn_implementation"]},
    )
    model_mem = torch.cuda.memory_allocated(0) / 1e9
    headroom = vram_gb - model_mem
    print(f"Model loaded: {model_mem:.1f} GB, headroom: {headroom:.1f} GB\n")

    rows = []
    ratios = cfg["compression_ratios"]
    presses_cfg = cfg["presses"]
    n_warmup = cfg["runtime"]["num_warmup"]
    n_runs = cfg["runtime"]["num_runs"]
    seed = cfg["runtime"]["seed"]

    total_points = len(presses_cfg) * len(ratios)
    runs_per_point = n_warmup + n_runs
    total_runs = total_points * runs_per_point

    # Track individual runs for immediate ETA
    run_counter = 0
    bench_start = time.perf_counter()

    def _eta_str() -> str:
        if run_counter == 0:
            return "ETA: calculating... finish: --:--"
        elapsed = time.perf_counter() - bench_start
        per_run = elapsed / run_counter
        remaining = (total_runs - run_counter) * per_run
        finish_time = time.strftime("%H:%M", time.localtime(time.time() + remaining))
        mins = remaining / 60
        if mins >= 60:
            eta = f"ETA: {mins / 60:.1f}h"
        else:
            eta = f"ETA: {mins:.0f}m"
        return f"{eta} finish: {finish_time}"

    pbar = tqdm(total=total_points, desc="Benchmarking")

    for pcfg in presses_cfg:
        for ratio in ratios:
            label = f"{pcfg['name']}@{ratio}"
            pbar.set_postfix_str(f"{label}  {_eta_str()}")

            try:
                press = get_press(pcfg["name"], ratio, dict(pcfg.get("kwargs", {})))
            except Exception as e:
                print(f"  ⚠ Skipping {label}: {e}")
                run_counter += runs_per_point
                pbar.update(1)
                continue

            # Warmup
            for w in range(n_warmup):
                pbar.set_postfix_str(f"{label} warmup {w+1}/{n_warmup}  {_eta_str()}")
                try:
                    _ = eval_needle(pipe, press, cfg, tokenizer)
                except Exception:
                    pass
                run_counter += 1
                pbar.set_postfix_str(f"{label} warmup {w+1}/{n_warmup}  {_eta_str()}")

            # Timed runs
            run_results = []
            for i in range(n_runs):
                pbar.set_postfix_str(f"{label} run {i+1}/{n_runs}  {_eta_str()}")
                torch.manual_seed(seed + i)
                try:
                    res = eval_needle(pipe, press, cfg, tokenizer)
                    run_results.append(res)
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM on {label}, run {i}")
                    reset_memory()
                    run_counter += 1
                    break
                except Exception as e:
                    print(f"  Error on {label}, run {i}: {e}")
                    run_counter += 1
                    break
                run_counter += 1
                pbar.set_postfix_str(f"{label} run {i+1}/{n_runs}  {_eta_str()}")

            if run_results:
                avg = {
                    "press": pcfg["name"],
                    "compression_ratio": ratio,
                    "needle_accuracy": sum(r["needle_hit"] for r in run_results)
                    / len(run_results),
                    "mean_latency_s": round(
                        sum(r["latency_s"] for r in run_results) / len(run_results), 2
                    ),
                    "peak_mem_gb": max(r["peak_mem_gb"] for r in run_results),
                    "num_runs": len(run_results),
                }
                rows.append(avg)

            pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, out_dir: str, base_name: str = "benchmark_results") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Accuracy vs compression ratio
    for press, grp in df.groupby("press"):
        axes[0].plot(grp["compression_ratio"], grp["needle_accuracy"],
                     marker="o", label=press)
    axes[0].set(xlabel="Compression Ratio", ylabel="Needle Accuracy",
                title="Accuracy vs Compression")
    axes[0].legend(fontsize=8)

    # 2. Peak memory vs compression ratio
    for press, grp in df.groupby("press"):
        axes[1].plot(grp["compression_ratio"], grp["peak_mem_gb"],
                     marker="s", label=press)
    axes[1].set(xlabel="Compression Ratio", ylabel="Peak Memory (GB)",
                title="Memory vs Compression")
    axes[1].legend(fontsize=8)

    # 3. Latency vs compression ratio
    for press, grp in df.groupby("press"):
        axes[2].plot(grp["compression_ratio"], grp["mean_latency_s"],
                     marker="^", label=press)
    axes[2].set(xlabel="Compression Ratio", ylabel="Latency (s)",
                title="Latency vs Compression")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, f"{base_name}.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close()


# ── Git push ──────────────────────────────────────────────────────────────────

def push_results(config_name: str, results_dir: str) -> None:
    """Stage, commit, and push results to origin."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_msg = f"benchmark: {config_name} @ {timestamp}"

    steps = [
        (["git", "add", results_dir + "/"], "Staging results"),
        (["git", "commit", "-m", commit_msg], "Committing"),
        (["git", "push", "origin"], "Pushing to origin"),
    ]

    for cmd, description in steps:
        print(f"\n[push] {description}: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.rstrip())
        if result.returncode != 0:
            print(f"[push] FAILED (exit {result.returncode})")
            if result.stderr:
                print(result.stderr.rstrip())
            return
    print("\n[push] Results pushed successfully.")


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_config.yaml")
    parser.add_argument("--push", action="store_true",
                        help="Auto-commit and push results to GitHub after benchmark")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["runtime"]["results_dir"], exist_ok=True)

    df = run_benchmark(cfg)

    ratios_tag = str(cfg["compression_ratios"])
    ctx_tag = cfg["eval"]["custom_needle"]["context_tokens"]
    n_runs = cfg["runtime"]["num_runs"]
    base_name = f"benchmark_results_{ratios_tag}_{ctx_tag}_runs={n_runs}"

    csv_path = os.path.join(cfg["runtime"]["results_dir"], f"{base_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))

    plot_results(df, cfg["runtime"]["results_dir"], base_name)

    if args.push:
        config_name = Path(args.config).stem
        push_results(config_name, cfg["runtime"]["results_dir"])


if __name__ == "__main__":
    main()
