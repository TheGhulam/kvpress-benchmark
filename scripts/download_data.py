#!/usr/bin/env python3
"""Download model weights and evaluation datasets for the kvpress benchmark.

Usage:
    python scripts/download_data.py                    # everything
    python scripts/download_data.py --model-only       # just the model
    python scripts/download_data.py --datasets-only    # just RULER + LooGLE
"""

import argparse
import os


def download_model(model_name: str) -> None:
    from huggingface_hub import snapshot_download

    print(f"\n{'='*60}")
    print(f"Downloading model: {model_name}")
    print(f"{'='*60}\n")

    # Windows often cannot create symlinks without Developer Mode/admin.
    # Keep the warning disabled by default unless user explicitly opts in.
    if os.name == "nt":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snapshot_download(
        repo_id=model_name,
        repo_type="model",
        cache_dir=cache_dir,
    )
    print(f"Model cached at: {cache_dir}")


def download_ruler() -> None:
    from datasets import get_dataset_config_names, load_dataset

    print("\n── Downloading RULER evaluation suite ──")
    configs = get_dataset_config_names("THUDM/LongBench", trust_remote_code=True)
    print(f"  Found {len(configs)} LongBench configs")

    downloaded = 0
    for config in configs:
        try:
            ds = load_dataset(
                "THUDM/LongBench",
                config,
                split="test",
                trust_remote_code=True,
            )
            print(f"  LongBench/{config}: {len(ds)} examples")
            downloaded += 1
        except Exception as e:
            print(f"  LongBench/{config}: skipped ({e})")

    print(f"  LongBench complete: {downloaded}/{len(configs)} configs cached")


def download_loogle() -> None:
    from datasets import load_dataset

    print("\n── Downloading LooGLE ──")
    for split in ["shortdep_qa", "longdep_qa", "longdep_summarization",
                   "shortdep_cloze"]:
        try:
            ds = load_dataset("bigainlp/LooGLE", split, trust_remote_code=True)
            test_split = ds["test"] if isinstance(ds, dict) else ds
            print(f"  LooGLE/{split}: {len(test_split)} examples")
        except Exception as e:
            print(f"  LooGLE/{split}: skipped ({e})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download data for kvpress benchmark")
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--datasets-only", action="store_true")
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID (default: Llama-3.1-8B-Instruct)",
    )
    args = parser.parse_args()

    if not args.datasets_only:
        download_model(args.model_name)

    if not args.model_only:
        download_ruler()
        download_loogle()

    print("\n✓ All downloads complete.\n")


if __name__ == "__main__":
    main()
