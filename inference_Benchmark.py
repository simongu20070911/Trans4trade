#!/usr/bin/env python3

import time
import torch
import torch.nn as nn

# Assuming all your model code + get_model() is in a file called models.py
# Adjust the import path as needed
from models import get_model

def measure_inference_time(model, input_data, n_runs=50, warm_up=10):
    """
    Measures the average inference time for 'n_runs' forward passes.
    A 'warm_up' number of forward passes is done first (ignored in timing)
    to allow any initial overhead (e.g., JIT warmup, GPU scheduling) to settle.
    """
    # Warm-up
    for _ in range(warm_up):
        _ = model(input_data)

    # Timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(input_data)
    end_time = time.time()

    # Average time per inference in milliseconds
    avg_time_ms = (end_time - start_time) / n_runs * 1000
    return avg_time_ms

def main():
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define a list of all model names available in your factory
    model_names = [
        "pure_attention",
        "s4transformer",         # Requires S4 libraries
        "lstm",
        "deeplob",
        "lstm_attn",
        "vanilla_transformer",
        "stacklstm",
        "trans_enc_lstm"
    ]

    # Basic hyperparameters for testing
    B, L, F = 32, 100, 128   # batch_size, seq_len, feature_dim
    input_data = torch.randn(B, L, F).to(device)

    # Number of forward passes to measure
    n_runs = 180
    warm_up = 20

    # Loop over each model
    for model_name in model_names:
        print("=" * 60)
        print(f"Model name: {model_name}")
        try:
            # Instantiate the model with typical hyperparams
            # Adjust as necessary (e.g. d_model=64, nhead=4, num_layers=2, etc.)
            model = get_model(
                model_name,
                input_size=F,
                output_size=1,
                d_model=64,
                nhead=4,
                num_layers=2,
                dropout=0.1,
                # additional optional arguments as needed...
            )

            # Move to device
            model.to(device)
            model.eval()

            # Count total parameters
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  -> Total trainable parameters: {total_params}")

            # Measure inference in eager mode
            eager_time_ms = measure_inference_time(model, input_data, n_runs, warm_up)
            print(f"  -> Eager mode average time: {eager_time_ms:.3f} ms")
            if 1==2: 
            # Measure inference in compiled mode (PyTorch 2.x+ only)
                if hasattr(torch, "compile"):
                    compiled_model = torch.compile(model)
                    compiled_time_ms = measure_inference_time(compiled_model, input_data, n_runs, warm_up)
                    print(f"  -> Compiled mode average time: {compiled_time_ms:.3f} ms")
                else:
                    print("  -> torch.compile() not available. Skipping compiled mode.")

        except Exception as e:
            # For models that might fail (e.g., S4 dependencies not installed)
            print(f"  !! Could not instantiate or run {model_name} due to error:\n{e}")

if __name__ == "__main__":
    main()
