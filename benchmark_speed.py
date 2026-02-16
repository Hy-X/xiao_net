import torch
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from models.xn_xiao_net_v5b_sigmoid import XiaoNetV5B

try:
    import seisbench.models as sbm
    HAS_SEISBENCH = True
except ImportError:
    HAS_SEISBENCH = False
    print("SeisBench not found. Skipping PhaseNet comparison.")

def benchmark_model(model, input_tensor, n_warmup=50, n_runs=100, device='cpu'):
    model.eval()
    
    # Warmup (critical for GPU to wake up and compile kernels)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)
    
    # Synchronize before starting timer (if GPU)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.time()
            _ = model(input_tensor)
            # Synchronize after execution to measure actual completion
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000) # Convert to ms
            
    return np.mean(times), np.std(times)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on: {device}")
    print("-" * 60)
    
    # Configuration
    batch_size = 1 # Edge inference typically uses batch size 1
    window_len = 3001
    dummy_input = torch.randn(batch_size, 3, window_len).to(device)
    
    results = []
    
    # 1. XiaoNetV5B Variants
    for bc in [8, 16, 32]:
        print(f"Benchmarking XiaoNetV5B (bc={bc})...")
        model = XiaoNetV5B(base_channels=bc).to(device)
        mean, std = benchmark_model(model, dummy_input, device=device)
        results.append((f"XiaoNetV5B (bc={bc})", mean, std))
    
    # 2. PhaseNet (Baseline)
    phasenet_mean = None
    if HAS_SEISBENCH:
        try:
            print("Benchmarking PhaseNet (Standard)...")
            phasenet = sbm.PhaseNet.from_pretrained("stead").to(device)
            mean_pn, std_pn = benchmark_model(phasenet, dummy_input, device=device)
            results.append(("PhaseNet", mean_pn, std_pn))
            phasenet_mean = mean_pn
        except Exception as e:
            print(f"Could not load PhaseNet: {e}")

    # Print Table
    print("\n" + "=" * 65)
    print(f"{'Model':<25} | {'Latency (ms)':<15} | {'Speedup vs PhaseNet':<20}")
    print("-" * 65)
    
    for name, mean, std in results:
        speedup = "-"
        if phasenet_mean and name != "PhaseNet":
            factor = phasenet_mean / mean
            speedup = f"{factor:.1f}x"
            
        print(f"{name:<25} | {mean:.2f} ± {std:.2f}      | {speedup:<20}")
    print("=" * 65)

if __name__ == "__main__":
    main()