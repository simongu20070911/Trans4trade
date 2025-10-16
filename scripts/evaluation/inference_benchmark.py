#!/usr/bin/env python3

"""Benchmark inference latency across registered models."""

from __future__ import annotations

import argparse
import time
from typing import Iterable

import torch

from trans4trade.models import get_model

def measure_inference_time(model: torch.nn.Module, input_data: torch.Tensor, *, n_runs: int = 50, warm_up: int = 10) -> float:
    """
    Measures the average inference time for 'n_runs' forward passes.
    A 'warm_up' number of forward passes is done first (ignored in timing)
    to allow any initial overhead (e.g., JIT warmup, GPU scheduling) to settle.
    """
    # Warm-up
    for _ in range(warm_up):
        _ = model(input_data)

    # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(input_data)
    end = time.time()
    return (end - start) / n_runs * 1000


def benchmark_models(
    model_names: Iterable[str],
    *,
    batch_size: int,
    seq_len: int,
    feature_dim: int,
    runs: int,
    warmup: int,
    use_compile: bool,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    input_data = torch.randn(batch_size, seq_len, feature_dim, device=device)

    for name in model_names:
        print("=" * 60)
        print(f"Model name: {name}")
        try:
            model = get_model(
                name,
                input_size=feature_dim,
                output_size=1,
                d_model=64,
                nhead=4,
                num_layers=2,
                dropout=0.1,
            )
            model.to(device)
            model.eval()

            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  -> Total trainable parameters: {total_params}")

            eager_time_ms = measure_inference_time(model, input_data, n_runs=runs, warm_up=warmup)
            print(f"  -> Eager mode average time: {eager_time_ms:.3f} ms")

            if use_compile and hasattr(torch, "compile"):
                compiled_model = torch.compile(model)
                compiled_time_ms = measure_inference_time(
                    compiled_model, input_data, n_runs=runs, warm_up=warmup
                )
                print(f"  -> Compiled mode average time: {compiled_time_ms:.3f} ms")
            elif use_compile:
                print("  -> torch.compile() not available. Skipping compiled mode.")

        except Exception as exc:  # noqa: BLE001
            print(f"  !! Could not instantiate or run {name} due to error:\n{exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "pure_attention",
            "s4transformer",
            "lstm",
            "deeplob",
            "lstm_attn",
            "vanilla_transformer",
            "stacklstm",
            "trans_enc_lstm",
        ],
        help="Model names registered in trans4trade.models.get_model",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--runs", type=int, default=180)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--compile", action="store_true", help="Benchmark with torch.compile if available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_models(
        args.models,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        feature_dim=args.features,
        runs=args.runs,
        warmup=args.warmup,
        use_compile=args.compile,
    )


if __name__ == "__main__":
    main()
