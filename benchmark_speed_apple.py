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

def get_apple_device():
    """
    Returns the best available device for Apple hardware.
    Prioritizes MPS (Metal Performance Shaders) for Apple Silicon.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        print("MPS not available. Using CPU.")
        return torch.device("cpu")

def benchmark_model(model, input_tensor, n_warmup=50, n_runs=100, device='cpu'):
    """
    Benchmark inference latency for a model with Apple Silicon optimizations.
    """
    model.eval()
    
    # Warmup (critical for MPS to initialize and compile Metal kernels)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)
    
    # Synchronize before starting timer
    if device.type == 'mps':
        # MPS synchronization - ensure all operations complete
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
        
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()  # Use perf_counter for higher precision
            _ = model(input_tensor)
            
            # Synchronize after execution to measure actual completion
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
                
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
            
    return np.mean(times), np.std(times), np.min(times), np.max(times)

def get_device_info(device):
    """Get information about the device being used."""
    if device.type == 'mps':
        return "Apple Silicon (Metal Performance Shaders)"
    elif device.type == 'cuda':
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        return "CPU"

def main():
    # Use Apple Silicon MPS if available
    device = get_apple_device()
    device_info = get_device_info(device)
    
    print("=" * 70)
    print(f"APPLE DEVICE BENCHMARK")
    print("=" * 70)
    print(f"Device: {device_info}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")
    print("-" * 70)
    
    # Configuration
    batch_size = 1  # Edge inference typically uses batch size 1
    window_len = 3001
    dummy_input = torch.randn(batch_size, 3, window_len).to(device)
    
    results = []
    
    # 1. XiaoNetV5B Variants
    for bc in [8, 16, 32]:
        print(f"Benchmarking XiaoNetV5B (bc={bc})...")
        model = XiaoNetV5B(base_channels=bc).to(device)
        mean, std, min_time, max_time = benchmark_model(model, dummy_input, device=device)
        results.append((f"XiaoNetV5B (bc={bc})", mean, std, min_time, max_time))
    
    # 2. PhaseNet (Baseline)
    phasenet_mean = None
    if HAS_SEISBENCH:
        try:
            print("Benchmarking PhaseNet (Standard)...")
            phasenet = sbm.PhaseNet.from_pretrained("stead").to(device)
            mean_pn, std_pn, min_pn, max_pn = benchmark_model(phasenet, dummy_input, device=device)
            results.append(("PhaseNet", mean_pn, std_pn, min_pn, max_pn))
            phasenet_mean = mean_pn
        except Exception as e:
            print(f"Could not load PhaseNet: {e}")

    # Print Table
    print("\n" + "=" * 85)
    print(f"{'Model':<25} | {'Mean (ms)':<12} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'Speedup':<10}")
    print("-" * 85)
    
    for name, mean, std, min_time, max_time in results:
        speedup = "-"
        if phasenet_mean and name != "PhaseNet":
            factor = phasenet_mean / mean
            speedup = f"{factor:.2f}x"
            
        print(f"{name:<25} | {mean:>6.2f}±{std:<4.2f} | {min_time:>8.2f} | {max_time:>8.2f} | {speedup:<10}")
    
    print("=" * 85)
    
    # Performance Tips
    print("\n" + "=" * 70)
    print("APPLE SILICON OPTIMIZATION TIPS:")
    print("-" * 70)
    print("1. Ensure PyTorch >= 1.12 for MPS support")
    print("2. Use 'torch.float32' for best MPS performance")
    print("3. Keep data on MPS device to avoid CPU↔GPU transfers")
    print("4. For production, consider exporting to Core ML (.mlmodel)")
    print("5. Batch processing may not improve speed on MPS (use batch=1)")
    print("=" * 70)

if __name__ == "__main__":
    main()
