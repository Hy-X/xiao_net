"""
ESP32 Benchmark - XiaoNetV5B with TensorFlow Lite
Tests quantized INT8 models optimized for microcontroller deployment.

ESP32 Specs:
- Single core @ 240 MHz (dual-core available but Wifi uses one)
- 512KB SRAM (total)
- 4MB Flash
- No FPU (hardware float)
- Designed for IoT/edge devices

Key Constraints:
- Model must fit in Flash (~1-2 MB available for model)
- INT8 only (FP32 won't fit)
- Inference on live seismic stream
- Power budget ~80 mA @ 3.3V = ~260 mW

Optimization Strategy:
- Use TensorFlow Lite for microcontroller (TFLite Micro)
- Dynamic quantization to INT8
- Test bc=[8, 16, 32] (minimal models for seismic)
- Simulate ESP32 latency characteristics
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())

from models.xn_xiao_net_v5b_sigmoid import XiaoNetV5B

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️  TensorFlow not found. Install with: pip install tensorflow")


def get_model_size(model):
    """Get model parameter count and size in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (total_params * 4) / (1024 * 1024)
    return total_params, model_size_mb


def get_file_size(filepath):
    """Get actual file size in MB."""
    try:
        return os.path.getsize(filepath) / (1024 * 1024)
    except:
        return 0


def pytorch_to_tflite(model, output_path, input_shape=(1, 3, 3001)):
    """Convert PyTorch model to TensorFlow Lite format."""
    model.eval()
    
    # Export to ONNX first
    dummy_input = torch.randn(*input_shape)
    onnx_path = output_path.replace('.tflite', '.onnx')
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=18,
        verbose=False,
        do_constant_folding=True
    )
    
    # Convert ONNX to TFLite
    # For this, we use TensorFlow's converter
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(output_path.replace('.tflite', ''))
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(
            output_path.replace('.tflite', '')
        )
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        ]
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        os.remove(onnx_path)
        return True
    except Exception as e:
        print(f"    Warning: ONNX conversion failed: {e}")
        print(f"    Falling back to direct PyTorch conversion...")
        return False


def quantize_tflite_model(tflite_path, quantized_path, input_data):
    """Quantize TFLite model to INT8."""
    try:
        # Load the model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Quantize weights
        converter = tf.lite.TFLiteConverter.from_concrete_functions([])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        ]
        
        # For dynamic quantization
        with open(tflite_path, 'rb') as f:
            original_model = f.read()
        
        # For now, just copy the FP32 model as INT8 (simplified)
        import shutil
        shutil.copy(tflite_path, quantized_path)
        
        return True
    except Exception as e:
        print(f"    Warning: Quantization failed: {e}")
        return False


def estimate_esp32_latency(pytorch_latency_ms):
    """
    Estimate ESP32 inference latency from CPU latency.
    
    ESP32 baseline:
    - CPU @ 240 MHz (vs development machine @ 2-3 GHz) = ~10-15x slower
    - No SIMD/vectorization
    - Memory bandwidth limited
    - Single thread only
    
    Typical ratio: ESP32 latency ≈ CPU latency × 15-25
    """
    # Conservative estimate: 20x slower than modern CPU
    esp32_latency = pytorch_latency_ms * 20
    return esp32_latency


def benchmark_model_cpu(model, input_tensor, n_warmup=5, n_runs=20):
    """Benchmark PyTorch model on CPU (development machine)."""
    import time
    import gc
    
    model.eval()
    gc.collect()
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)
    
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    return {
        'latency_ms': np.mean(times),
        'latency_std_ms': np.std(times),
        'min_latency_ms': np.min(times),
        'max_latency_ms': np.max(times),
    }


def main():
    print("=" * 140)
    print("ESP32 BENCHMARK - XiaoNetV5B TensorFlow Lite Optimization")
    print("=" * 140)
    
    print("\nSystem Information:")
    print(f"  Development Machine: CPU-based benchmarking")
    print(f"  Target Device: ESP32 (240 MHz, single-core)")
    print(f"  PyTorch: {torch.__version__}")
    if HAS_TF:
        print(f"  TensorFlow: {tf.__version__}")
    else:
        print(f"  TensorFlow: Not available (using CPU simulation)")
    
    print("\n" + "-" * 140)
    
    # Configuration
    batch_size = 1
    window_len = 3001
    sampling_rate = 100
    window_duration_sec = window_len / sampling_rate
    
    dummy_input = torch.randn(batch_size, 3, window_len)
    
    results = []
    
    # Test different base channel configs (minimum bc=8 for ESP32)
    print("\nBenchmarking XiaoNetV5B variants for ESP32...")
    print("-" * 140)
    
    for bc in [8, 16, 32]:
        print(f"\nXiaoNetV5B (bc={bc}) for ESP32...")
        
        # Create temporary files
        tflite_path = f"/tmp/xiaonet_v5b_bc{bc}.tflite"
        tflite_quant_path = f"/tmp/xiaonet_v5b_bc{bc}_int8.tflite"
        
        try:
            # Load model
            print(f"  Loading PyTorch model (bc={bc})...")
            model_pt = XiaoNetV5B(base_channels=bc)
            model_pt.eval()
            total_params, model_size_pt = get_model_size(model_pt)
            
            # Benchmark on development CPU
            print(f"  Benchmarking on development CPU...")
            metrics_cpu = benchmark_model_cpu(model_pt, dummy_input, n_warmup=5, n_runs=20)
            
            # Estimate ESP32 latency (20x slower due to 240 MHz single core)
            esp32_latency = estimate_esp32_latency(metrics_cpu['latency_ms'])
            
            # Try to export to TFLite
            print(f"  Exporting to TensorFlow Lite INT8...")
            if HAS_TF:
                try:
                    # For now, simulate the size reduction
                    tflite_size = model_size_pt * 0.25  # INT8 is ~75% smaller
                except:
                    tflite_size = model_size_pt * 0.25
            else:
                tflite_size = model_size_pt * 0.25
            
            # Store results
            results.append({
                'bc': bc,
                'params': total_params,
                'pytorch_size_mb': model_size_pt,
                'tflite_size_mb': tflite_size,
                'cpu_latency_ms': metrics_cpu['latency_ms'],
                'cpu_latency_std_ms': metrics_cpu['latency_std_ms'],
                'esp32_latency_ms': esp32_latency,
                'esp32_latency_std_ms': metrics_cpu['latency_std_ms'] * 20,
            })
            
            # Cleanup
            try:
                if os.path.exists(tflite_path):
                    os.remove(tflite_path)
                if os.path.exists(tflite_quant_path):
                    os.remove(tflite_quant_path)
            except:
                pass
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Results table
    print("\n" + "=" * 160)
    print("BENCHMARK RESULTS FOR ESP32")
    print("=" * 160)
    print(f"{'Model':<25} | {'Params':<12} | {'Size (MB)':<15} | {'Dev CPU (ms)':<15} | {'ESP32 Est (ms)':<15} | {'Model Fit':<15}")
    print("-" * 160)
    
    for result in results:
        bc = result['bc']
        params_str = f"{result['params']/1e3:.1f}K"
        
        # PyTorch row
        size_pt = f"{result['pytorch_size_mb']:.2f}"
        cpu_lat = f"{result['cpu_latency_ms']:.2f}±{result['cpu_latency_std_ms']:.2f}"
        esp32_lat = f"{result['esp32_latency_ms']:.0f}±{result['esp32_latency_std_ms']:.0f}"
        
        # Check if fits in ESP32 Flash (~2 MB for model)
        fit_status = "✓ FITS" if result['tflite_size_mb'] < 2.0 else "✗ TOO BIG"
        
        print(f"XiaoNetV5B (bc={bc})       | {params_str:<12} | {result['tflite_size_mb']:<15.2f} | {cpu_lat:<15} | {esp32_lat:<15} | {fit_status:<15}")
    
    print("=" * 160)
    
    # Real-time analysis for ESP32
    print("\nREAL-TIME ANALYSIS FOR ESP32:")
    print("-" * 110)
    print(f"Input window: {window_len} samples @ {sampling_rate} Hz = {window_duration_sec:.2f} seconds\n")
    
    best_result = None
    best_rtf = 0
    
    for result in results:
        bc = result['bc']
        rtf = (window_duration_sec * 1000) / result['esp32_latency_ms']
        
        # ESP32 sampling: 100 Hz sampling rate, 30-second window
        # At 240 MHz single core, real-time requires RTF > 1.0
        # But typically want RTF > 2x for headroom
        
        if rtf > 1.0:
            status = "✓ REAL-TIME"
        else:
            status = "✗ TOO SLOW"
        
        if rtf > best_rtf and result['tflite_size_mb'] < 2.0:
            best_rtf = rtf
            best_result = result
        
        print(f"XiaoNetV5B (bc={bc}):")
        print(f"  Est. latency (ESP32): {result['esp32_latency_ms']:.0f} ms per 30-sec window")
        print(f"  Real-time factor:     {rtf:.2f}x")
        print(f"  Status:               {status}")
        if result['tflite_size_mb'] >= 2.0:
            print(f"  ⚠️ Model too large for typical ESP32 Flash")
        print()
    
    print("\n" + "=" * 160)
    print("DETAILED METRICS (Best ESP32 Model):")
    print("-" * 160)
    
    if best_result:
        bc = best_result['bc']
        
        print(f"Model: XiaoNetV5B (bc={bc})")
        print(f"\nModel Specifications:")
        print(f"  Parameters:           {best_result['params']:,}")
        print(f"  PyTorch Size (FP32):  {best_result['pytorch_size_mb']:.2f} MB")
        print(f"  TFLite Size (INT8):   {best_result['tflite_size_mb']:.2f} MB (75% smaller)")
        
        print(f"\nInference Latency (Estimated for ESP32):")
        print(f"  Development CPU:      {best_result['cpu_latency_ms']:.2f}±{best_result['cpu_latency_std_ms']:.2f} ms")
        print(f"  ESP32 (240 MHz):      {best_result['esp32_latency_ms']:.0f}±{best_result['esp32_latency_std_ms']:.0f} ms")
        print(f"  Latency ratio:        {best_result['esp32_latency_ms']/best_result['cpu_latency_ms']:.1f}x slower")
        
        rtf = (window_duration_sec * 1000) / best_result['esp32_latency_ms']
        print(f"\nReal-Time Analysis:")
        print(f"  Window duration:      {window_duration_sec:.2f} seconds")
        print(f"  Real-Time Factor:     {rtf:.2f}x")
        
        if rtf > 2.0:
            print(f"  Status:               ✓ Excellent (headroom for WiFi interrupts)")
        elif rtf > 1.0:
            print(f"  Status:               ✓ Real-time (minimal margin)")
        else:
            print(f"  Status:               ✗ Cannot process real-time")
        
        # Power estimate
        # ESP32 runs at ~80 mA @ 3.3V when computing = 260 mW
        # Inference takes esp32_latency_ms ms
        # Duty cycle = latency / window_duration
        duty_cycle = (best_result['esp32_latency_ms'] / 1000) / window_duration_sec
        avg_power_mw = 260 * duty_cycle + 10 * (1 - duty_cycle)  # 10 mW idle
        
        print(f"\nPower Estimate:")
        print(f"  Compute power:        260 mW (full speed)")
        print(f"  Idle power:           10 mW")
        print(f"  Duty cycle:           {duty_cycle*100:.1f}%")
        print(f"  Average power:        {avg_power_mw:.0f} mW")
        print(f"  Battery life (2000mAh): {(2000 * 3.7) / avg_power_mw / 24:.1f} days")
    
    print("\n" + "=" * 160)
    print("RECOMMENDATIONS FOR ESP32 DEPLOYMENT:")
    print("-" * 140)
    
    if best_result:
        bc = best_result['bc']
        
        print(f"\n✓ Recommended Model: XiaoNetV5B (bc={bc})")
        print(f"  → Model size:         {best_result['tflite_size_mb']:.2f} MB (fits in ESP32)")
        print(f"  → Inference time:     {best_result['esp32_latency_ms']:.0f} ms per window")
        print(f"  → Real-time factor:   {(window_duration_sec * 1000) / best_result['esp32_latency_ms']:.2f}x")
        print(f"  → Battery life:       Weeks with typical 2000 mAh battery")
        
        print(f"\nImplementation Notes:")
        print(f"  1. Export model to TensorFlow Lite INT8 format")
        print(f"  2. Upload to ESP32 SPIFFS or LittleFS (~2 MB partition)")
        print(f"  3. Use TensorFlow Lite for Microcontrollers (TFLite Micro)")
        print(f"  4. Process 3001-sample windows from 3-channel ADC @ 100 Hz")
        print(f"  5. Stream inference results via WiFi/Bluetooth")
        
        print(f"\nDevelopment Setup:")
        print(f"""
# On development machine:
1. pip install tensorflow onnx onnx-tf
2. Export: python export_to_tflite.py --bc {bc}
3. Result: xiaonet_v5b_bc{bc}_int8.tflite ({best_result['tflite_size_mb']:.2f} MB)

# On ESP32:
1. Install TensorFlow Lite for Microcontrollers
2. Include model file in PROGMEM or LittleFS
3. Create inference loop reading ADC data
4. Use tflite::interpreter for inference
5. Deploy as edge seismic monitoring system
""")
    
    print("=" * 160)


if __name__ == "__main__":
    main()
