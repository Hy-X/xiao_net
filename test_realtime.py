"""
Real-time streaming simulation test for XiaoNetV5B
Simulates continuous seismic data streaming and checks if inference keeps up.
"""

import torch
import time
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from models.xn_xiao_net_v5b_sigmoid import XiaoNetV5B

def test_streaming_realtime(model_bc=16, num_windows=100, sampling_rate=100, window_len=3001):
    """
    Simulate continuous streaming and measure if inference can keep up.
    
    Args:
        model_bc: Base channels for XiaoNetV5B
        num_windows: Number of windows to process
        sampling_rate: Sampling rate in Hz
        window_len: Window length in samples
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Real-Time Streaming Test on: {device}")
    print("=" * 80)
    
    # Setup
    model = XiaoNetV5B(base_channels=model_bc).to(device)
    model.eval()
    
    window_duration_ms = (window_len / sampling_rate) * 1000  # milliseconds
    
    print(f"Config:")
    print(f"  Model: XiaoNetV5B (bc={model_bc})")
    print(f"  Windows to process: {num_windows}")
    print(f"  Window size: {window_len} samples @ {sampling_rate} Hz = {window_duration_ms:.2f} ms")
    print(f"  Total duration: {num_windows * window_duration_ms / 1000:.2f} seconds")
    print()
    
    # Warmup
    print("Warming up model...")
    dummy_input = torch.randn(1, 3, window_len).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Streaming simulation
    print(f"Processing {num_windows} windows...\n")
    
    inference_times = []
    real_clock_times = []
    
    expected_time_ms = 0  # Expected time if processing at real-time speed
    
    with torch.no_grad():
        for window_idx in range(num_windows):
            # Simulate data arrival (without actual data I/O)
            input_data = torch.randn(1, 3, window_len).to(device)
            
            # Measure inference
            real_clock_start = time.time()
            _ = model(input_data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            real_clock_end = time.time()
            
            inference_ms = (real_clock_end - real_clock_start) * 1000
            real_clock_times.append(inference_ms)
            inference_times.append(inference_ms)
            expected_time_ms += window_duration_ms
            
            # Progress report every 10 windows
            if (window_idx + 1) % 10 == 0:
                total_inference = sum(inference_times)
                rtf = expected_time_ms / total_inference if total_inference > 0 else 0
                print(f"  Window {window_idx+1:3d}: {inference_ms:6.2f} ms | "
                      f"Cumulative RTF: {rtf:5.1f}x | "
                      f"Avg: {np.mean(inference_times):.2f} ms")
    
    # Analysis
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("-" * 80)
    
    total_inference_time = sum(inference_times)
    avg_inference = np.mean(inference_times)
    std_inference = np.std(inference_times)
    min_inference = np.min(inference_times)
    max_inference = np.max(inference_times)
    
    rtf = expected_time_ms / total_inference_time if total_inference_time > 0 else 0
    
    print(f"Total data duration:        {expected_time_ms/1000:.2f} seconds")
    print(f"Total inference time:       {total_inference_time:.2f} ms")
    print(f"Real-Time Factor (RTF):     {rtf:.1f}x")
    print()
    print(f"Inference per window:")
    print(f"  Average:  {avg_inference:.2f} ms")
    print(f"  Std Dev:  {std_inference:.2f} ms")
    print(f"  Min:      {min_inference:.2f} ms")
    print(f"  Max:      {max_inference:.2f} ms")
    print()
    
    # Real-time verdict
    if rtf >= 1.0:
        headroom_pct = (rtf - 1.0) * 100
        print(f"✓ CAN RUN REAL-TIME with {headroom_pct:.1f}% headroom for I/O and other tasks")
    else:
        deficit_pct = (1.0 - rtf) * 100
        print(f"✗ CANNOT RUN REAL-TIME - {deficit_pct:.1f}% too slow")
    
    # Recommendation
    max_latency_budget = window_duration_ms  # Full window duration is budget
    if max_inference < max_latency_budget * 0.1:
        print(f"✓ Excellent headroom: max inference ({max_inference:.2f}ms) << budget ({max_latency_budget:.2f}ms)")
    elif max_inference < max_latency_budget * 0.5:
        print(f"✓ Good headroom: max inference ({max_inference:.2f}ms) < budget ({max_latency_budget:.2f}ms)")
    else:
        print(f"⚠ Low headroom: max inference ({max_inference:.2f}ms) is close to budget ({max_latency_budget:.2f}ms)")
    
    print("=" * 80)
    
    return rtf >= 1.0, rtf


def test_batch_streaming(model_bc=16, batch_size=10, num_batches=10):
    """
    Test if batching multiple windows improves throughput.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nBatch Streaming Test")
    print("=" * 80)
    
    model = XiaoNetV5B(base_channels=model_bc).to(device)
    model.eval()
    
    window_len = 3001
    sampling_rate = 100
    window_duration_ms = (window_len / sampling_rate) * 1000
    
    print(f"Config:")
    print(f"  Model: XiaoNetV5B (bc={model_bc})")
    print(f"  Batch size: {batch_size} windows")
    print(f"  Number of batches: {num_batches}")
    print()
    
    # Warmup
    dummy_batch = torch.randn(batch_size, 3, window_len).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    batch_times = []
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            batch_data = torch.randn(batch_size, 3, window_len).to(device)
            
            start = time.time()
            _ = model(batch_data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            
            batch_ms = (end - start) * 1000
            batch_times.append(batch_ms)
            
            # Per-window latency in this batch
            per_window_latency = batch_ms / batch_size
            data_duration = window_duration_ms * batch_size
            rtf_batch = data_duration / batch_ms
            
            print(f"  Batch {batch_idx+1}: {batch_ms:.2f} ms | "
                  f"Per-window: {per_window_latency:.2f} ms | RTF: {rtf_batch:.1f}x")
    
    avg_batch_time = np.mean(batch_times)
    per_window_avg = avg_batch_time / batch_size
    data_duration = window_duration_ms * batch_size
    rtf_overall = data_duration / avg_batch_time
    
    print()
    print(f"Average batch time:      {avg_batch_time:.2f} ms")
    print(f"Per-window latency:      {per_window_avg:.2f} ms")
    print(f"Overall RTF:             {rtf_overall:.1f}x")
    print("=" * 80)
    
    return rtf_overall


if __name__ == "__main__":
    # Test single-window streaming
    passing, rtf = test_streaming_realtime(model_bc=16, num_windows=100)
    
    # Test batching
    batch_rtf = test_batch_streaming(model_bc=16, batch_size=10, num_batches=20)
    
    print("\nSummary:")
    print(f"  Single-window streaming RTF: {rtf:.1f}x - {'PASS ✓' if passing else 'FAIL ✗'}")
    print(f"  Batch (10x) streaming RTF:   {batch_rtf:.1f}x")
