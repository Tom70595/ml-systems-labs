import torch
import time
import statistics
import sys
sys.path.append('lab03/src')
from models import MLP, CNN, AttentionModel
from thop import profile
from torchinfo import summary
import io
from contextlib import redirect_stdout

def get_params(model):
    """Use torchinfo for accurate parameter counts (thop misses Transformer internals)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        summary(model, input_size=(1, 1, 28, 28), verbose=1)
    # Extract total params from summary string
    for line in buf.getvalue().split("\n"):
        if line.startswith("Total params:"):
            return int(line.split(":")[1].strip().replace(",", ""))
    return 0

def measure_flops(model, input_size=(1, 1, 28, 28)):
    model.eval()
    dummy_input = torch.randn(*input_size)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return flops

def measure_latency(model, batch_size, runs):
    model.eval()
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    with torch.no_grad():
        model(dummy_input)
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(dummy_input)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    return statistics.median(times)

if __name__ == "__main__":
    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel()
    }
    
    print(f"{'Model':<12} | {'Params':<10} | {'FLOPs (M)':<10} | {'Lat B=1 (ms)':<14} | {'Lat B=128 (ms)':<14}")
    print("-" * 70)
    
    results = {}
    
    for name, model in models.items():
        params = get_params(model)
        flops = measure_flops(model)
        flops_m = flops / 1e6
        lat_1 = measure_latency(model, batch_size=1, runs=100)
        lat_128 = measure_latency(model, batch_size=128, runs=20)
        
        results[name] = {
            "params": params,
            "flops_m": flops_m,
            "lat_1": lat_1,
            "lat_128": lat_128
        }
        
        print(f"{name:<12} | {params:<10,} | {flops_m:<10.2f} | {lat_1:<14.4f} | {lat_128:<14.4f}")

    import csv, os
    os.makedirs("lab03/results", exist_ok=True)
    with open("lab03/results/cost_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "flops_millions", "latency_b1_ms", "latency_b128_ms"])
        for name, r in results.items():
            w.writerow([name, r["params"], f"{r['flops_m']:.2f}", f"{r['lat_1']:.4f}", f"{r['lat_128']:.4f}"])
    
    print("\nSaved to lab03/results/cost_metrics.csv")