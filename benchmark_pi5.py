"""
Raspberry Pi 5 Benchmark for XiaoNetV5B
Optimized for ARM64 CPU-only inference with thermal monitoring.

Pi 5 Specs:
- 4x ARM Cortex-A76 @ 2.4 GHz
- 4GB or 8GB LPDDR5
- No GPU (CPU-only)
- Thermal throttling at ~80°C
- Limited to ~10W continuous

Key differences from GPU benchmarking:
- CPU prefers larger batches (better cache utilization)
- Threading matters (torch.set_num_threads)
- Thermal throttling is critical to monitor
- Power consumption affects real-world usability
"""

import torch
import time
import numpy as np
import sys
import os
import psutil
import gc
import subprocess
from pathlib import Path

sys.path.append(os.getcwd())

from models.xn_xiao_net_v5b_sigmoid import XiaoNetV5B

try:
    import seisbench.models as sbm
    HAS_SEISBENCH = True
except ImportError:
    HAS_SEISBENCH = False
    print("⚠️  SeisBench not found. Skipping PhaseNet comparison.")

def get_cpu_frequency():
    """Get current CPU frequency in GHz (Pi 5 specific)."""
    try:
        # Read from /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
            freq_khz = int(f.read().strip())
            return freq_khz / 1e6  # Convert to GHz
    except:
        return None

def get_cpu_temp():
    """Get CPU temperature in Celsius (Pi 5 specific)."""
    try:
        # Read from /sys/class/thermal/thermal_zone0/temp
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp_millidegrees = int(f.read().strip())
            return temp_millidegrees / 1000  # Convert to Celsius
    except:
        return None

def get_cpu_throttle_status():
    """Check if CPU is being throttled (Pi 5 specific)."""
    try:
        # Check vcgencmd if available
        result = subprocess.run(['vcgencmd', 'get_throttled'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            throttled = result.stdout.strip()
            # Format: throttled=0x50000 (bit 4 = throttled, bit 16 = has throttled)
            return throttled
    except:
        return None

def get_memory_info():
    """Get detailed memory information."""
    mem = psutil.virtual_memory()
    return {
        'total_mb': mem.total / (1024**2),
        'available_mb': mem.available / (1024**2),
        'used_mb': mem.used / (1024**2),
        'percent': mem.percent
    }

def get_model_size(model):
    """Get model parameter count and size in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    # Assuming 32-bit floats (4 bytes per parameter)
    model_size_mb = (total_params * 4) / (1024 * 1024)
    return total_params, model_size_mb

def benchmark_model_cpu(model, input_tensor, n_warmup=10, n_runs=50):
    """
    Benchmark CPU inference with thermal and frequency monitoring.
    
    Pi 5 optimization notes:
    - Fewer warmup runs (CPU less critical than GPU)
    - Fewer measurements (to avoid thermal throttling during benchmark)
    - Monitor CPU frequency and temperature
    - Thread count should be set before calling this function
    """
    model.eval()
    
    # Force garbage collection
    gc.collect()
    
    # Record baseline conditions
    temp_before = get_cpu_temp()
    freq_before = get_cpu_frequency()
    mem_before = get_memory_info()
    
    print(f"    Initial: {temp_before:.1f}°C | {freq_before:.2f} GHz | {mem_before['used_mb']:.0f} MB RAM")
    
    # Warmup (fewer runs on Pi due to thermal sensitivity)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)
    
    # Small delay to stabilize
    time.sleep(0.5)
    
    # Measure inference times
    times = []
    temps = []
    freqs = []
    
    with torch.no_grad():
        for _ in range(n_runs):
            temp = get_cpu_temp()
            freq = get_cpu_frequency()
            
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # ms
            if temp:
                temps.append(temp)
            if freq:
                freqs.append(freq)
    
    # Record final conditions
    temp_after = get_cpu_temp()
    freq_after = get_cpu_frequency()
    mem_after = get_memory_info()
    mem_delta = mem_after['used_mb'] - mem_before['used_mb']
    
    # Analysis
    mean_time = np.mean(times)
    std_time = np.std(times)
    mean_temp = np.mean(temps) if temps else None
    max_temp = np.max(temps) if temps else None
    mean_freq = np.mean(freqs) if freqs else None
    min_freq = np.min(freqs) if freqs else None
    throttled = get_cpu_throttle_status()
    
    return {
        'latency_ms': mean_time,
        'latency_std_ms': std_time,
        'min_latency_ms': np.min(times),
        'max_latency_ms': np.max(times),
        'temp_before_c': temp_before,
        'temp_after_c': temp_after,
        'mean_temp_c': mean_temp,
        'max_temp_c': max_temp,
        'freq_before_ghz': freq_before,
        'freq_after_ghz': freq_after,
        'mean_freq_ghz': mean_freq,
        'min_freq_ghz': min_freq,
        'throttled': throttled,
        'mem_delta_mb': mem_delta,
        'iterations': n_runs
    }

def main():
    device = torch.device('cpu')
    print("=" * 90)
    print("RASPBERRY PI 5 BENCHMARK - XiaoNetV5B")
    print("=" * 90)
    
    # System info
    print("\nSystem Information:")
    print(f"  Device: {device}")
    num_threads = torch.get_num_threads()
    print(f"  CPU threads: {num_threads}")
    print(f"  RAM: {get_memory_info()['total_mb']:.0f} MB total, {get_memory_info()['available_mb']:.0f} MB available")
    
    temp = get_cpu_temp()
    freq = get_cpu_frequency()
    if temp:
        print(f"  CPU Temp: {temp:.1f}°C")
    if freq:
        print(f"  CPU Freq: {freq:.2f} GHz")
    
    print("\n" + "-" * 90)
    
    # Configuration
    batch_size = 1  # Pi 5 prefers small batches to reduce latency
    window_len = 3001
    sampling_rate = 100
    window_duration_sec = window_len / sampling_rate
    
    dummy_input = torch.randn(batch_size, 3, window_len)
    
    results = []
    
    # Test different base channel configs
    print("\nBenchmarking XiaoNetV5B variants...")
    print("-" * 90)
    
    for bc in [4, 8, 16, 32]:
        print(f"XiaoNetV5B (bc={bc})...")
        model = XiaoNetV5B(base_channels=bc).to(device)
        total_params, model_size_mb = get_model_size(model)
        
        # Benchmark with default thread count
        metrics = benchmark_model_cpu(model, dummy_input, n_warmup=5, n_runs=30)
        
        # Check thermal throttling
        throttle_warning = ""
        if metrics['throttled'] and '0x1' in metrics['throttled']:
            throttle_warning = " ⚠️ THROTTLED"
        elif metrics['max_temp_c'] and metrics['max_temp_c'] > 75:
            throttle_warning = " ⚠️ HOT"
        
        results.append({
            'name': f"XiaoNetV5B (bc={bc})",
            'params': total_params,
            'size_mb': model_size_mb,
            'metrics': metrics,
            'warning': throttle_warning
        })
        
        # Small cooldown between models
        time.sleep(1)
    
    # Benchmark PhaseNet baseline
    phasenet_latency = None
    if HAS_SEISBENCH:
        try:
            print("\nBenchmarking PhaseNet (baseline)...")
            phasenet = sbm.PhaseNet.from_pretrained("stead").to(device)
            total_params, model_size_mb = get_model_size(phasenet)
            
            metrics = benchmark_model_cpu(phasenet, dummy_input, n_warmup=5, n_runs=30)
            phasenet_latency = metrics['latency_ms']
            
            # Check thermal throttling
            throttle_warning = ""
            if metrics['throttled'] and '0x1' in metrics['throttled']:
                throttle_warning = " ⚠️ THROTTLED"
            elif metrics['max_temp_c'] and metrics['max_temp_c'] > 75:
                throttle_warning = " ⚠️ HOT"
            
            results.append({
                'name': 'PhaseNet',
                'params': total_params,
                'size_mb': model_size_mb,
                'metrics': metrics,
                'warning': throttle_warning,
                'is_baseline': True
            })
            
            time.sleep(1)
        except Exception as e:
            print(f"⚠️  Could not load PhaseNet: {e}")
    
    # Results table
    print("\n" + "=" * 150)
    print("BENCHMARK RESULTS")
    print("=" * 150)
    print(f"{'Model':<25} | {'Params':<12} | {'Size (MB)':<10} | {'Latency (ms)':<15} | {'Temp (°C)':<12} | {'Speedup':<10} | Status")
    print("-" * 150)
    
    for result in results:
        m = result['metrics']
        name = result['name']
        params_str = f"{result['params']/1e3:.1f}K"
        latency = f"{m['latency_ms']:.2f}±{m['latency_std_ms']:.2f}"
        temp = f"{m['mean_temp_c']:.1f}/{m['max_temp_c']:.1f}" if m['mean_temp_c'] else "N/A"
        status = result['warning'] if result['warning'] else "✓"
        
        # Calculate speedup
        speedup = "-"
        if phasenet_latency and not result.get('is_baseline', False):
            speedup_factor = phasenet_latency / m['latency_ms']
            speedup = f"{speedup_factor:.1f}x"
        
        print(f"{name:<25} | {params_str:<12} | {result['size_mb']:<10.2f} | {latency:<15} | {temp:<12} | {speedup:<10} | {status}")
    
    print("=" * 150)
    
    # Real-time analysis
    print("\nREAL-TIME ANALYSIS FOR PI 5:")
    print("-" * 110)
    print(f"Input window: {window_len} samples @ {sampling_rate} Hz = {window_duration_sec:.2f} seconds\n")
    
    best_result = None
    best_rtf = 0
    
    for result in results:
        m = result['metrics']
        rtf = (window_duration_sec * 1000) / m['latency_ms']
        
        status = "✓ REAL-TIME" if rtf > 1.0 else "✗ TOO SLOW"
        throttle_note = " (⚠️ THROTTLED)" if result['warning'] else ""
        
        # Speedup vs PhaseNet
        speedup_str = ""
        if phasenet_latency and not result.get('is_baseline', False):
            speedup_factor = phasenet_latency / m['latency_ms']
            speedup_str = f" | {speedup_factor:.1f}x faster"
        
        print(f"{result['name']:<25} | RTF: {rtf:>6.1f}x | {status}{throttle_note}{speedup_str}")
        
        if rtf > best_rtf and not result.get('is_baseline', False):
            best_rtf = rtf
            best_result = result
    
    print("\n" + "=" * 150)
    print("DETAILED METRICS (Best XiaoNetV5B performer):")
    print("-" * 150)
    
    if best_result:
        m = best_result['metrics']
        print(f"Model: {best_result['name']}")
        print(f"Threads: {num_threads} (system default)")
        print(f"\nLatency:")
        print(f"  Mean:     {m['latency_ms']:.2f} ms")
        print(f"  Std Dev:  {m['latency_std_ms']:.2f} ms")
        print(f"  Min:      {m['min_latency_ms']:.2f} ms")
        print(f"  Max:      {m['max_latency_ms']:.2f} ms")
        
        print(f"\nThermal:")
        print(f"  Before:   {m['temp_before_c']:.1f}°C")
        print(f"  After:    {m['temp_after_c']:.1f}°C")
        print(f"  Mean:     {m['mean_temp_c']:.1f}°C")
        print(f"  Max:      {m['max_temp_c']:.1f}°C")
        
        print(f"\nCPU Frequency:")
        print(f"  Before:   {m['freq_before_ghz']:.2f} GHz")
        print(f"  After:    {m['freq_after_ghz']:.2f} GHz")
        print(f"  Mean:     {m['mean_freq_ghz']:.2f} GHz")
        print(f"  Min:      {m['min_freq_ghz']:.2f} GHz")
        
        if m['throttled']:
            print(f"\nThrottling Status: {m['throttled']}")
        else:
            print(f"\nThrottling Status: None (✓ No throttling detected)")
        
        print(f"\nMemory Delta: {m['mem_delta_mb']:.1f} MB")
        print(f"Iterations: {m['iterations']}")
    
    print("=" * 150)
    
    # Recommendations
    print("\nRECOMMENDATIONS FOR PI 5 DEPLOYMENT:")
    print("-" * 90)
    if best_result:
        m = best_result['metrics']
        if phasenet_latency:
            speedup_vs_phasenet = phasenet_latency / m['latency_ms']
            print(f"✓ {best_result['name']} is {speedup_vs_phasenet:.1f}x faster than PhaseNet")
            print(f"  → PhaseNet: {phasenet_latency:.2f} ms")
            print(f"  → {best_result['name']}: {m['latency_ms']:.2f} ms")
            print(f"  → Saves {((phasenet_latency - m['latency_ms']) / phasenet_latency * 100):.1f}% latency\n")
        
        if m['max_temp_c'] > 75:
            print("⚠️  HIGH TEMPERATURE: Consider adding heatsink or fan to prevent thermal throttling")
        elif m['max_temp_c'] > 65:
            print("⚠️  ELEVATED TEMPERATURE: Monitor thermal performance. Consider passive cooling.")
        else:
            print("✓ Temperature within safe range")
        
        if m['throttled'] and '0x1' in m['throttled']:
            print("⚠️  THROTTLING DETECTED: CPU was throttled during benchmark")
            print("    → Use external cooling solution (heatsink + fan)")
            print("    → Process only when within thermal budget")
            print("    → Consider batch vs streaming tradeoffs")
        else:
            print("✓ No CPU throttling detected")
        
        print(f"\n✓ Recommended configuration: {best_result['name']}")
        print(f"  → Latency: {m['latency_ms']:.2f} ms per window")
        print(f"  → RTF: {(window_duration_sec * 1000) / m['latency_ms']:.1f}x")
        print(f"  → Can process continuous seismic stream on Pi 5")
        print(f"  → Model size: {best_result['size_mb']:.2f} MB")
        print(f"  → RAM overhead: {m['mem_delta_mb']:.1f} MB")
    
    print("=" * 150)


if __name__ == "__main__":
    main()
