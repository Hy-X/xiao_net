import argparse
import csv
import gc
import os
import sys
import time
from typing import List, Tuple, Dict

import numpy as np
import psutil
import torch

try:
    import onnx  # optional
    HAS_ONNX = True
except Exception:
    HAS_ONNX = False

try:
    import onnxruntime as ort
    HAS_ONNXRT = True
except Exception:
    HAS_ONNXRT = False

# Add project root to path so local models can be imported
sys.path.append(os.getcwd())

try:
    from models.xn_xiao_net_v5b_sigmoid import XiaoNetV5B
except Exception as e:
    raise

try:
    import seisbench.models as sbm
    HAS_SEISBENCH = True
except Exception:
    HAS_SEISBENCH = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XiaoNetV5B benchmark (latency, throughput, memory)")
    p.add_argument("--device", default=None, help="Device to use (cpu or cuda). Auto-detect if omitted")
    p.add_argument("--base-channels", type=int, nargs="+", default=[8, 16, 32],
                   help="List of base channel values for XiaoNetV5B to benchmark")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1], help="Batch sizes to test")
    p.add_argument("--window-lens", type=int, nargs="+", default=[3001], help="Input window lengths to test")
    p.add_argument("--sampling-rate", type=int, default=100, help="Sampling rate (Hz) used to compute RTF")
    p.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations")
    p.add_argument("--runs", type=int, default=200, help="Number of timed runs")
    p.add_argument("--save-csv", default=None, help="Path to save results as CSV")
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--export-onnx", action="store_true", help="Export each PyTorch model to ONNX")
    p.add_argument("--onnx-runtime", action="store_true", help="Run inference using ONNX Runtime (requires onnxruntime)")
    p.add_argument("--onnx-opset", type=int, default=11, help="ONNX opset for export")
    p.add_argument("--onnx-path", default=None, help="Directory or filename prefix for exported ONNX models")
    return p.parse_args()


def get_model_size(model: torch.nn.Module) -> Tuple[int, float]:
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (total_params * 4) / (1024 * 1024)
    return total_params, model_size_mb


def get_system_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_benchmark(model: torch.nn.Module, input_tensor: torch.Tensor, n_warmup: int, n_runs: int,
                  device: torch.device) -> Dict:
    model.eval()

    dtype = next(model.parameters()).dtype

    # Cleanup before measurement
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    system_mem_before = get_system_memory_mb()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(input_tensor)

    # Timed runs
    times = []
    with torch.inference_mode():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times = np.array(times)

    gpu_peak_mb = 0.0
    if device.type == "cuda":
        gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    system_mem_after = get_system_memory_mb()

    return {
        "latency_mean_ms": float(times.mean()),
        "latency_std_ms": float(times.std()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "gpu_peak_mb": float(gpu_peak_mb),
        "system_mem_delta_mb": float(system_mem_after - system_mem_before),
        "runs": int(n_runs),
    }


def format_params(params: int) -> str:
    return f"{params/1e6:.2f}M" if params >= 1e6 else f"{params/1e3:.1f}K"


def save_results_csv(path: str, rows: List[Dict]):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def export_to_onnx(model: torch.nn.Module, input_tensor: torch.Tensor, path: str, opset: int = 11):
    model.eval()
    # Ensure CPU export if CUDA tensors/types cause issues; export with current device if possible
    # Use example input as the tracing input
    export_input = input_tensor
    try:
        torch.onnx.export(
            model,
            export_input,
            path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size", 2: "time"}, "output": {0: "batch_size", 2: "time"}},
        )
        return True
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False


def run_onnxruntime(onnx_path: str, input_numpy: np.ndarray, warmup: int, runs: int, use_cuda: bool) -> Dict:
    if not HAS_ONNXRT:
        raise RuntimeError("onnxruntime not available")

    providers = ["CPUExecutionProvider"]
    if use_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        _ = sess.run(None, {input_name: input_numpy})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: input_numpy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times)

    return {
        "onnx_latency_mean_ms": float(times.mean()),
        "onnx_latency_std_ms": float(times.std()),
        "onnx_p50_ms": float(np.percentile(times, 50)),
        "onnx_p95_ms": float(np.percentile(times, 95)),
        "onnx_runs": int(runs),
        "onnx_provider": providers[0],
    }


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Benchmarking on: {device}")

    dtype = torch.float16 if args.dtype == "float16" and device.type == "cuda" else torch.float32

    results = []

    for bc in args.base_channels:
        for batch in args.batch_sizes:
            for window_len in args.window_lens:
                sampling_rate = args.sampling_rate
                window_duration_s = window_len / sampling_rate

                # Build model and input
                print(f"Building XiaoNetV5B (bc={bc}) for batch={batch}, window={window_len}...", flush=True)
                model = XiaoNetV5B(base_channels=bc).to(device=device, dtype=dtype)

                total_params, model_size_mb = get_model_size(model)

                dummy_input = torch.randn(batch, 3, window_len, device=device, dtype=dtype)

                stats = run_benchmark(model, dummy_input, args.warmup, args.runs, device)

                mean_ms = stats["latency_mean_ms"]
                throughput_inf_per_sec = 1000.0 / mean_ms
                samples_per_sec = throughput_inf_per_sec * batch

                row = {
                    "model": f"XiaoNetV5B_bc{bc}",
                    "base_channels": bc,
                    "batch_size": batch,
                    "window_len": window_len,
                    "sampling_rate": sampling_rate,
                    "window_duration_s": f"{window_duration_s:.4f}",
                    "params": total_params,
                    "params_human": format_params(total_params),
                    "model_size_mb": f"{model_size_mb:.2f}",
                    "latency_mean_ms": f"{mean_ms:.3f}",
                    "latency_std_ms": f"{stats['latency_std_ms']:.3f}",
                    "p50_ms": f"{stats['p50_ms']:.3f}",
                    "p95_ms": f"{stats['p95_ms']:.3f}",
                    "inferences_per_sec": f"{throughput_inf_per_sec:.2f}",
                    "samples_per_sec": f"{samples_per_sec:.1f}",
                    "gpu_peak_mb": f"{stats['gpu_peak_mb']:.1f}",
                    "system_mem_delta_mb": f"{stats['system_mem_delta_mb']:.1f}",
                }

                results.append(row)

                # Export and benchmark ONNX if requested
                if args.export_onnx:
                    onnx_dir = args.onnx_path if args.onnx_path else "."
                    os.makedirs(onnx_dir, exist_ok=True) if os.path.isdir(onnx_dir) or onnx_dir == "." else None
                    # Build filename
                    prefix = onnx_dir if args.onnx_path and os.path.isdir(onnx_dir) else os.path.join(onnx_dir)
                    onnx_name = f"xnv5b_bc{bc}_b{batch}_w{window_len}.onnx"
                    onnx_path = onnx_name if prefix == "." else os.path.join(prefix, onnx_name)

                    exported = False
                    if HAS_ONNX:
                        exported = export_to_onnx(model, dummy_input, onnx_path, opset=args.onnx_opset)
                        if exported:
                            row["onnx_path"] = onnx_path
                        else:
                            row["onnx_path"] = "export_failed"
                    else:
                        row["onnx_path"] = "onnx_missing"

                    if args.onnx_runtime and exported and HAS_ONNXRT:
                        # ONNX Runtime expects NHW or NCHW; our input is (B, C, T). We'll pass as-is.
                        input_np = dummy_input.cpu().numpy()
                        use_cuda = device.type == "cuda"
                        try:
                            onnx_stats = run_onnxruntime(onnx_path, input_np, args.warmup, args.runs, use_cuda)
                            row.update({
                                "onnx_latency_mean_ms": f"{onnx_stats['onnx_latency_mean_ms']:.3f}",
                                "onnx_latency_std_ms": f"{onnx_stats['onnx_latency_std_ms']:.3f}",
                                "onnx_p95_ms": f"{onnx_stats['onnx_p95_ms']:.3f}",
                                "onnx_provider": onnx_stats["onnx_provider"],
                            })
                        except Exception as e:
                            row.update({"onnx_runtime_error": str(e)})
                    elif args.onnx_runtime and not HAS_ONNXRT:
                        row.update({"onnx_runtime_error": "onnxruntime_not_installed"})

    # Optionally benchmark PhaseNet if available
    if HAS_SEISBENCH:
        try:
            print("Benchmarking PhaseNet from SeisBench as baseline...")
            phasenet = sbm.PhaseNet.from_pretrained("stead").to(device)
            total_params, model_size_mb = get_model_size(phasenet)
            dummy_input = torch.randn(1, 3, args.window_lens[0], device=device)
            stats = run_benchmark(phasenet, dummy_input, args.warmup, args.runs, device)
            mean_ms = stats["latency_mean_ms"]
            row = {
                "model": "PhaseNet",
                "base_channels": None,
                "batch_size": 1,
                "window_len": args.window_lens[0],
                "sampling_rate": args.sampling_rate,
                "window_duration_s": f"{args.window_lens[0]/args.sampling_rate:.4f}",
                "params": total_params,
                "params_human": format_params(total_params),
                "model_size_mb": f"{model_size_mb:.2f}",
                "latency_mean_ms": f"{mean_ms:.3f}",
                "latency_std_ms": f"{stats['latency_std_ms']:.3f}",
                "p50_ms": f"{stats['p50_ms']:.3f}",
                "p95_ms": f"{stats['p95_ms']:.3f}",
                "inferences_per_sec": f"{1000.0/mean_ms:.2f}",
                "samples_per_sec": f"{1000.0/mean_ms * 1:.1f}",
                "gpu_peak_mb": f"{stats['gpu_peak_mb']:.1f}",
                "system_mem_delta_mb": f"{stats['system_mem_delta_mb']:.1f}",
            }
            results.append(row)
        except Exception as e:
            print(f"PhaseNet benchmark skipped: {e}")

    # Print a compact table
    print("\nResults:")
    header = ["model", "batch_size", "window_len", "params_human", "model_size_mb", "latency_mean_ms", "inferences_per_sec", "samples_per_sec", "gpu_peak_mb"]
    print(" | ".join(header))
    for r in results:
        print(" | ".join(str(r[h]) for h in header))

    if args.save_csv:
        save_results_csv(args.save_csv, results)
        print(f"Saved results to {args.save_csv}")


if __name__ == "__main__":
    main()