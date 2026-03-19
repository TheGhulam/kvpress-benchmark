#!/usr/bin/env python3
"""Quick memory profile for a single press + compression ratio.

Usage:
    python scripts/profile_memory.py
    python scripts/profile_memory.py --press SnapKVPress --ratio 0.3
    python scripts/profile_memory.py --context-length 65536
"""

import argparse
import gc
import time

import torch


def profile(model_name: str, press_name: str, ratio: float,
            context_length: int, dtype_str: str) -> None:
    import kvpress
    from transformers import pipeline as hf_pipeline, AutoTokenizer

    dtype = getattr(torch, dtype_str)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = hf_pipeline(
        "kv-press-text-generation",
        model=model_name,
        device_map="auto",
        torch_dtype=dtype,
        model_kwargs={"attn_implementation": "sdpa"},
    )

    # Build a long dummy context
    filler = "The quick brown fox jumps over the lazy dog. "
    filler_ids = tokenizer(filler, add_special_tokens=False)["input_ids"]
    repeats = context_length // len(filler_ids) + 1
    context = filler * repeats
    context_ids = tokenizer(context, add_special_tokens=False,
                            max_length=context_length, truncation=True)
    prompt = tokenizer.decode(context_ids["input_ids"]) + "\nSummarize the above."

    press_cls = getattr(kvpress, press_name)
    press = press_cls(compression_ratio=ratio)

    # Reset and measure
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"\n{'='*60}")
    print(f"Model:        {model_name}")
    print(f"Press:        {press_name} @ {ratio}")
    print(f"Context:      ~{context_length} tokens")
    print(f"Dtype:        {dtype_str}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()
    out = pipe(
        prompt,
        press=press,
        max_new_tokens=128,
        do_sample=False,
        return_full_text=False,
    )
    elapsed = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 1e9
    if isinstance(out, list) and out:
        first = out[0]
        answer = first.get("generated_text", "") if isinstance(first, dict) else str(first)
    elif isinstance(out, dict):
        answer = out.get("generated_text") or out.get("answer") or ""
    else:
        answer = str(out)

    print(torch.cuda.memory_summary(abbreviated=True))
    print(f"\nPeak memory:  {peak:.2f} GB")
    print(f"Wall time:    {elapsed:.1f}s")
    print(f"Answer:       {answer[:200]}...\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--press", default="ExpectedAttentionPress")
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--context-length", type=int, default=16384)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    profile(args.model, args.press, args.ratio, args.context_length, args.dtype)


if __name__ == "__main__":
    main()
