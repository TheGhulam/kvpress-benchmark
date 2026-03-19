#!/usr/bin/env python3
"""KV cache quantization × pruning stacking benchmark.

Extends the base kvpress benchmark to answer:
  1. Does int4/int2 KV quantization + pruning provide additive memory savings?
  2. Does quantization degrade accuracy when combined with aggressive pruning?
  3. Can the combination enable 16K context on 48 GB VRAM (the OOM barrier)?
  4. What is the quality/memory Pareto frontier?

Sweeps every (press, compression_ratio, kv_nbits) triple from the config.

Results are saved as CSV + heatmaps + Pareto plots in the results/ directory.

Usage:
    python scripts/benchmark_quant_stack.py
    python scripts/benchmark_quant_stack.py --config configs/quant_stack.yaml
"""

import argparse
import gc
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
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

    if "inner_press" in kwargs:
        inner_name = kwargs.pop("inner_press")
        inner_cls = getattr(kvpress, inner_name)
        inner = inner_cls(compression_ratio=compression_ratio)
        return cls(press=inner, **kwargs)

    if name == "ThinKPress":
        return cls(key_channel_compression_ratio=compression_ratio, **kwargs)

    return cls(compression_ratio=compression_ratio, **kwargs)


def get_cache(nbits: int, model_config=None):
    """Return a QuantizedCache or None (DynamicCache) for the given bit-width."""
    if nbits == 0 or nbits == 16:
        return None

    from transformers import QuantizedCache
    return QuantizedCache(
        "quanto",
        model_config,
        nbits=nbits,
        q_group_size=64,
        residual_length=128,
    )

def measure_memory() -> dict:
    """Return peak GPU memory stats in GB (device 0 only)."""
    allocated = torch.cuda.max_memory_allocated(0) / 1e9
    reserved = torch.cuda.max_memory_reserved(0) / 1e9
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

    needle_query_tokens = len(
        tokenizer(needle + " " + query, add_special_tokens=False)["input_ids"]
    )
    num_fillers = (context_tokens - needle_query_tokens) // tokens_per_filler
    half = num_fillers // 2

    context = (filler_sentence * half) + f"\n{needle}\n" + (filler_sentence * half)
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"


def eval_needle(pipe, press, cache, cfg: dict, tokenizer) -> dict:
    """Run the custom needle-in-a-haystack test with optional quantized cache."""
    ncfg = cfg["eval"]["custom_needle"]
    prompt = build_needle_prompt(
        ncfg["context_tokens"], ncfg["needle"], ncfg["query"], tokenizer
    )

    reset_memory()
    t0 = time.perf_counter()
    pipe_kwargs = dict(max_new_tokens=64, do_sample=False, return_full_text=False)
    if press is not None:
        pipe_kwargs["press"] = press
    if cache is not None:
        pipe_kwargs["cache"] = cache
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
    import kvpress
    from transformers import pipeline as hf_pipeline, AutoTokenizer

    model_name = cfg["model"]["name"]
    dtype = getattr(torch, cfg["model"]["dtype"])

    print(f"\nLoading model: {model_name}  (dtype={cfg['model']['dtype']})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    # Resolve model config for QuantizedCache
    model_config = getattr(pipe.model, "config", None)
    if model_config is None:
        # Some pipelines nest the model one level deeper
        inner = getattr(pipe, "model", None)
        for attr in ("model", "transformer", "base_model"):
            candidate = getattr(inner, attr, None)
            if candidate is not None and hasattr(candidate, "config"):
                model_config = candidate.config
                break
    if model_config is None:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(model_name)
    print(f"Resolved model config: {type(model_config).__name__}")

    rows = []
    ratios = cfg["compression_ratios"]
    presses_cfg = cfg["presses"]
    kv_nbits_list = cfg["kv_quantization"]["nbits_list"]
    n_warmup = cfg["runtime"]["num_warmup"]
    n_runs = cfg["runtime"]["num_runs"]
    seed = cfg["runtime"]["seed"]

    # Include a "no press" baseline for each quantization level
    all_press_configs = [{"name": "NoPrune", "kwargs": {}}] + presses_cfg
    # For NoPrune, only ratio 0.0 makes sense
    total_points = 0
    for pcfg in all_press_configs:
        for nbits in kv_nbits_list:
            if pcfg["name"] == "NoPrune":
                total_points += 1  # single ratio = 0.0
            else:
                total_points += len(ratios)

    runs_per_point = n_warmup + n_runs
    total_runs = total_points * runs_per_point

    run_counter = 0
    bench_start = time.perf_counter()

    def _eta_str() -> str:
        if run_counter == 0:
            return "ETA: calculating..."
        elapsed = time.perf_counter() - bench_start
        per_run = elapsed / run_counter
        remaining = (total_runs - run_counter) * per_run
        finish_time = time.strftime("%H:%M", time.localtime(time.time() + remaining))
        mins = remaining / 60
        eta = f"ETA: {mins / 60:.1f}h" if mins >= 60 else f"ETA: {mins:.0f}m"
        return f"{eta} finish: {finish_time}"

    pbar = tqdm(total=total_points, desc="Benchmarking")

    for pcfg in all_press_configs:
        for nbits in kv_nbits_list:
            active_ratios = [0.0] if pcfg["name"] == "NoPrune" else ratios
            for ratio in active_ratios:
                quant_label = f"int{nbits}" if nbits > 0 and nbits < 16 else "fp16"
                label = f"{pcfg['name']}@{ratio}+{quant_label}"
                pbar.set_postfix_str(f"{label}  {_eta_str()}")

                # --- Instantiate press ---
                try:
                    if pcfg["name"] == "NoPrune":
                        press = None
                    else:
                        press = get_press(pcfg["name"], ratio,
                                          dict(pcfg.get("kwargs", {})))
                except Exception as e:
                    print(f"  ⚠ Skipping {label}: {e}")
                    run_counter += runs_per_point
                    pbar.update(1)
                    continue

                # --- Instantiate cache ---
                try:
                    cache = get_cache(nbits, model_config)
                except Exception as e:
                    print(f"  ⚠ Cache error for {label}: {e}")
                    run_counter += runs_per_point
                    pbar.update(1)
                    continue

                # Warmup
                for w in range(n_warmup):
                    pbar.set_postfix_str(
                        f"{label} warmup {w+1}/{n_warmup}  {_eta_str()}")
                    try:
                        _ = eval_needle(pipe, press, cache, cfg, tokenizer)
                    except Exception:
                        pass
                    run_counter += 1

                # Timed runs
                run_results = []
                for i in range(n_runs):
                    pbar.set_postfix_str(
                        f"{label} run {i+1}/{n_runs}  {_eta_str()}")
                    torch.manual_seed(seed + i)
                    try:
                        # Need a fresh cache object for each run
                        cache = get_cache(nbits, model_config)
                        res = eval_needle(pipe, press, cache, cfg, tokenizer)
                        run_results.append(res)
                    except torch.cuda.OutOfMemoryError:
                        print(f"  OOM on {label}, run {i}")
                        reset_memory()
                        run_counter += 1
                        # Record OOM as a data point
                        run_results.append({
                            "needle_hit": -1,
                            "answer": "OOM",
                            "latency_s": -1,
                            "peak_mem_gb": -1,
                            "peak_reserved_gb": -1,
                            "vram_total_gb": -1,
                        })
                        break
                    except Exception as e:
                        print(f"  Error on {label}, run {i}: {e}")
                        run_counter += 1
                        break
                    run_counter += 1

                if run_results:
                    valid = [r for r in run_results if r["needle_hit"] >= 0]
                    oom_count = sum(1 for r in run_results if r["needle_hit"] < 0)
                    avg = {
                        "press": pcfg["name"],
                        "compression_ratio": ratio,
                        "kv_nbits": nbits if nbits > 0 else 16,
                        "kv_quant_label": quant_label,
                        "needle_accuracy": (
                            sum(r["needle_hit"] for r in valid) / len(valid)
                            if valid else 0.0
                        ),
                        "mean_latency_s": round(
                            sum(r["latency_s"] for r in valid) / len(valid), 2
                        ) if valid else -1,
                        "peak_mem_gb": (
                            max(r["peak_mem_gb"] for r in valid)
                            if valid else -1
                        ),
                        "num_runs": len(valid),
                        "oom_count": oom_count,
                    }
                    rows.append(avg)

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, out_dir: str, context_tokens: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")

    # Filter out NoPrune rows for the main sweep plots
    df_press = df[df["press"] != "NoPrune"].copy()
    df_baseline = df[df["press"] == "NoPrune"].copy()

    # ── Figure 1: Accuracy heatmap (press × ratio) per kv_nbits ──────────
    nbits_vals = sorted(df_press["kv_nbits"].unique())
    fig, axes = plt.subplots(1, len(nbits_vals), figsize=(7 * len(nbits_vals), 6),
                             squeeze=False)
    for ax, nbits in zip(axes[0], nbits_vals):
        sub = df_press[df_press["kv_nbits"] == nbits].copy()
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index="press", columns="compression_ratio",
            values="needle_accuracy", aggfunc="mean"
        )
        sns.heatmap(pivot, annot=True, fmt=".0%", cmap="RdYlGn", vmin=0, vmax=1,
                    ax=ax, cbar_kws={"label": "Accuracy"})
        quant_label = f"int{nbits}" if nbits < 16 else "fp16 (baseline)"
        ax.set_title(f"Accuracy — KV cache {quant_label}\n({context_tokens//1024}K ctx)")
        ax.set_ylabel("Press")
        ax.set_xlabel("Compression Ratio")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_accuracy_heatmap.png"), dpi=150)
    plt.close()
    print(f"  Saved fig1_accuracy_heatmap.png")

    # ── Figure 2: Memory comparison (fp16 vs int4 vs int2) ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: memory vs compression ratio, grouped by kv_nbits
    for nbits in nbits_vals:
        sub = df_press[df_press["kv_nbits"] == nbits]
        # Average across presses for clarity
        avg_by_ratio = sub.groupby("compression_ratio")["peak_mem_gb"].mean()
        label = f"int{nbits}" if nbits < 16 else "fp16"
        axes[0].plot(avg_by_ratio.index, avg_by_ratio.values, marker="o", label=label)

    # Add baselines (NoPrune)
    for _, row in df_baseline.iterrows():
        label = f"NoPrune int{int(row['kv_nbits'])}" if row["kv_nbits"] < 16 else "NoPrune fp16"
        axes[0].axhline(y=row["peak_mem_gb"], linestyle="--", alpha=0.5,
                        label=label)

    axes[0].set(xlabel="Compression Ratio", ylabel="Peak Memory (GB)",
                title=f"Memory vs Compression — avg across presses ({context_tokens//1024}K ctx)")
    axes[0].legend(fontsize=8)

    # Right: per-press memory at ratio 0.5
    ratio_mid = 0.5
    sub = df_press[df_press["compression_ratio"] == ratio_mid].copy()
    if not sub.empty:
        sub["label"] = sub["press"] + " + " + sub["kv_quant_label"]
        sub = sub.sort_values("peak_mem_gb")
        colors = ["#2ecc71" if n < 16 else "#3498db" for n in sub["kv_nbits"]]
        axes[1].barh(sub["label"], sub["peak_mem_gb"], color=colors)
        axes[1].set(xlabel="Peak Memory (GB)",
                    title=f"Memory @ ratio={ratio_mid} ({context_tokens//1024}K ctx)")
        axes[1].axvline(x=48, color="red", linestyle="--", alpha=0.7, label="48 GB VRAM")
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_memory_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved fig2_memory_comparison.png")

    # ── Figure 3: Latency comparison ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for nbits in nbits_vals:
        sub = df_press[df_press["kv_nbits"] == nbits]
        avg_by_ratio = sub.groupby("compression_ratio")["mean_latency_s"].mean()
        label = f"int{nbits}" if nbits < 16 else "fp16"
        ax.plot(avg_by_ratio.index, avg_by_ratio.values, marker="^", label=label)
    ax.set(xlabel="Compression Ratio", ylabel="Mean Latency (s)",
           title=f"Latency vs Compression ({context_tokens//1024}K ctx)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_latency_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved fig3_latency_comparison.png")

    # ── Figure 4: Pareto frontier (accuracy vs memory) ───────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {"fp16": "o", "int4": "s", "int2": "D"}
    for _, row in df_press.iterrows():
        if row["peak_mem_gb"] < 0:
            continue
        m = markers.get(row["kv_quant_label"], "x")
        ax.scatter(row["peak_mem_gb"], row["needle_accuracy"],
                   marker=m, s=80, alpha=0.7,
                   label=f"{row['press']}+{row['kv_quant_label']}@{row['compression_ratio']}")

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Too many labels — just show quant type markers
    ax.axvline(x=48, color="red", linestyle="--", alpha=0.7)
    ax.text(48.5, 0.5, "48 GB VRAM limit", color="red", fontsize=9)
    ax.set(xlabel="Peak Memory (GB)", ylabel="Needle Accuracy",
           title=f"Pareto: Accuracy vs Memory ({context_tokens//1024}K ctx)")

    # Add simplified legend for marker types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="gray", label="fp16 KV", markersize=8,
               linestyle="None"),
        Line2D([0], [0], marker="s", color="gray", label="int4 KV", markersize=8,
               linestyle="None"),
        Line2D([0], [0], marker="D", color="gray", label="int2 KV", markersize=8,
               linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_pareto_frontier.png"), dpi=150)
    plt.close()
    print(f"  Saved fig4_pareto_frontier.png")


# ── Multi-context sweep ──────────────────────────────────────────────────────

def run_context_sweep(cfg: dict) -> pd.DataFrame:
    """Run the benchmark at each configured context length, concatenating results."""
    context_lengths = cfg["eval"]["custom_needle"].get("context_tokens_sweep", None)
    if context_lengths is None:
        context_lengths = [cfg["eval"]["custom_needle"]["context_tokens"]]

    all_dfs = []
    for ctx_len in context_lengths:
        print(f"\n{'='*60}")
        print(f"  Context length: {ctx_len:,} tokens")
        print(f"{'='*60}")
        cfg["eval"]["custom_needle"]["context_tokens"] = ctx_len
        df = run_benchmark(cfg)
        df["context_tokens"] = ctx_len
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


# ── Git push ──────────────────────────────────────────────────────────────────

def push_results(config_name: str, results_dir: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_msg = f"quant-stack benchmark: {config_name} @ {timestamp}"
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
    parser = argparse.ArgumentParser(
        description="KV cache quantization × pruning stacking benchmark")
    parser.add_argument("--config", default="configs/quant_stack.yaml")
    parser.add_argument("--push", action="store_true",
                        help="Auto-commit and push results after benchmark")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = cfg["runtime"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    df = run_context_sweep(cfg)

    # Save combined CSV
    csv_path = os.path.join(results_dir, "quant_stack_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))

    # Generate per-context-length plots
    for ctx_len in df["context_tokens"].unique():
        sub = df[df["context_tokens"] == ctx_len]
        ctx_dir = os.path.join(results_dir, f"ctx_{ctx_len}")
        os.makedirs(ctx_dir, exist_ok=True)
        plot_results(sub, ctx_dir, ctx_len)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Best configurations per context length")
    print("=" * 70)
    for ctx_len in sorted(df["context_tokens"].unique()):
        sub = df[(df["context_tokens"] == ctx_len) & (df["needle_accuracy"] == 1.0)]
        if sub.empty:
            print(f"\n  {ctx_len//1024}K: No configuration achieved 100% accuracy")
            # Show best
            sub = df[df["context_tokens"] == ctx_len].nlargest(3, "needle_accuracy")
            for _, r in sub.iterrows():
                print(f"    {r['press']}+int{int(r['kv_nbits'])}@{r['compression_ratio']}"
                      f"  acc={r['needle_accuracy']:.0%}  mem={r['peak_mem_gb']:.1f}GB")
        else:
            best = sub.nsmallest(1, "peak_mem_gb").iloc[0]
            print(f"\n  {ctx_len//1024}K: Best = {best['press']}+int{int(best['kv_nbits'])}"
                  f"@{best['compression_ratio']}  "
                  f"mem={best['peak_mem_gb']:.1f}GB  lat={best['mean_latency_s']:.1f}s")

    if args.push:
        config_name = Path(args.config).stem
        push_results(config_name, results_dir)


if __name__ == "__main__":
    main()
