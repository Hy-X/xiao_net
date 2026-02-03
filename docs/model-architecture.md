# XiaoNet Model Architectures

This document describes the five XiaoNet model variants, their design principles, and use cases.

## Overview

XiaoNet provides a family of progressively optimized U-Net architectures for seismic phase picking, ranging from a baseline model to edge-optimized variants suitable for INT8 quantization and deployment on resource-constrained devices.

All models share:
- **U-Net encoder-decoder** structure with skip connections
- **3-channel input** (Z, N, E seismic components)
- **3-class output** (P-wave, S-wave, Noise probabilities)
- **Softmax activation** for mutually exclusive phase predictions
- **Configurable window length** (default 1000 samples at 100 Hz = 10 seconds)

---

## Model Variants

| Model | File | Parameters | Target Use Case |
|-------|------|-----------|-----------------|
| **v1 (Original)** | [xn_xiao_net.py](../models/xn_xiao_net.py) | ~8K | Baseline student model |
| **v2** | [xn_xiao_net_v2.py](../models/xn_xiao_net_v2.py) | ~15K | Speed-optimized baseline |
| **v3** | [xn_xiao_net_v3.py](../models/xn_xiao_net_v3.py) | ~4K | Balanced speed/accuracy |
| **v4 (XiaoNetFast)** | [xn_xiao_net_v4.py](../models/xn_xiao_net_v4.py) | ~3K | CPU inference, real-time |
| **v5 (XiaoNetEdge)** | [xn_xiao_net_v5.py](../models/xn_xiao_net_v5.py) | ~1.5K | Edge deployment, INT8 quantization |

---

## Detailed Comparison

### Architecture Components

| Feature | v2 | v3 | v4 | v5 |
|---------|----|----|----|----|
| **Base Channels** | 8 | 6 | 6 | 4 |
| **Bottleneck Width** | base×8 (64 ch) | base×8 (48 ch) | base×6 (36 ch) | base×6 (24 ch) |
| **Kernel Size** | 7 | 3 | 3 | 3 |
| **Encoder Type** | Standard Conv1d | Depthwise Separable | Depthwise Separable | Depthwise Separable |
| **Decoder Upsampling** | ConvTranspose1d | ConvTranspose1d | Upsample + Conv1d | Upsample + Conv1d |
| **Activation** | ReLU | ReLU | ReLU | Hardswish |
| **Decoder BatchNorm** | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| **Skip Connections** | Additive | Additive | Additive | FloatFunctional (quantization-ready) |
| **Softmax Layer** | nn.Softmax | nn.Softmax | nn.Softmax | F.softmax (functional) |

### Performance Characteristics

| Model | Model Size (FP32) | Inference Speed | Memory Usage | Edge Deployment |
|-------|------------------|-----------------|--------------|-----------------|
| **v2** | ~60 KB | Moderate | Medium | ⚠️ Acceptable |
| **v3** | ~16 KB | Fast | Low | ✅ Good |
| **v4** | ~12 KB | Faster | Lower | ✅ Very Good |
| **v5** | ~6 KB | Fastest | Lowest | ✅ Optimal |

---

## Design Evolution

### v2: Speed-Optimized Baseline

**Key Features:**
- Aggressive early downsampling (stride=4) reduces computation
- ConvTranspose1d for learned upsampling
- Additive skip connections (no channel concatenation)
- Kernel size 7 for better receptive field

**Best For:** Initial experiments, accuracy baseline

```python
from models.xn_xiao_net_v2 import XiaoNet

model = XiaoNet(
    window_len=1000,
    in_channels=3,
    num_phases=3,
    base_channels=8
)
```

---

### v3: Balanced Middle Ground

**Key Improvements over v2:**
- Depthwise separable convolutions (fewer parameters, faster)
- Reduced kernel size (7→3) for lower latency
- Base channels reduced (8→6) for smaller model

**Best For:** Balanced accuracy and speed, general-purpose deployment

**Trade-offs:**
- ⚠️ ConvTranspose1d may have poor support on some NPUs
- ⚠️ ReLU with `inplace=True` breaks quantization

```python
from models.xn_xiao_net_v3 import XiaoNet

model = XiaoNet(
    window_len=1000,
    in_channels=3,
    num_phases=3,
    base_channels=6
)
```

---

### v4: Real-Time CPU Inference

**Key Improvements over v3:**
- Upsample + Conv1d replaces ConvTranspose1d (faster, better NPU support)
- Removed BatchNorm from decoder (fewer operations)
- Narrower bottleneck (×8→×6) reduces compute

**Best For:** Real-time CPU inference, streaming applications

```python
from models.xn_xiao_net_v4 import XiaoNetFast

model = XiaoNetFast(
    window_len=1000,
    in_channels=3,
    num_phases=3,
    base_channels=6
)
```

---

### v5: Edge and INT8 Deployment

**Key Improvements over v4:**
- **Hardswish activation:** Better for quantization, common in MobileNet/EfficientNet
- **FloatFunctional skip connections:** Quantization-aware skip addition
- **No inplace operations:** Enables static quantization
- **Smallest base channels (4):** Minimal memory footprint
- **Functional softmax:** Explicit about FP32 post-processing

**Best For:** Edge devices (Raspberry Pi, Jetson Nano), INT8 quantization, embedded systems

**Quantization Support:**
- Static quantization: 2-4× CPU speedup with INT8
- ONNX export ready for edge runtimes
- Compatible with TensorFlow Lite/ONNX Runtime

```python
from models.xn_xiao_net_v5 import XiaoNetEdge

model = XiaoNetEdge(
    window_len=1000,
    in_channels=3,
    num_phases=3,
    base_channels=4
)

# Export to ONNX for deployment
import torch
dummy_input = torch.randn(1, 3, 1000)
torch.onnx.export(
    model, 
    dummy_input, 
    "xiaonet_edge.onnx",
    opset_version=13,
    input_names=['waveform'],
    output_names=['phase_probs']
)
```

---

## Architecture Details

### Encoder Structure (All Variants)

```
Input (batch, 3, 1000)
    ↓
enc1: Conv → BN → Activation
    (stride=1, keeps temporal resolution)
    ↓
enc2: Conv → BN → Activation  
    (stride=4, downsample to 250 samples)
    ↓
enc3: Conv → BN → Activation
    (stride=4, downsample to ~62 samples)
    ↓
bottleneck: Conv → BN → Activation
    (stride=4, downsample to ~15 samples)
```

### Decoder Structure (All Variants)

```
bottleneck output (~15 samples)
    ↓
dec3: Upsample → Conv → [BN] → Activation
    + skip from enc3 (additive)
    ↓
dec2: Upsample → Conv → [BN] → Activation
    + skip from enc2 (additive)
    ↓
dec1: Upsample → Conv → [BN] → Activation
    + skip from enc1 (additive)
    ↓
output: Conv1d (1×1) → Softmax
    ↓
Output (batch, 3, 1000)
```

**Note:** `[BN]` indicates BatchNorm is only present in v2 and v3.

---

## Depthwise Separable Convolutions

Used in v3, v4, and v5 to reduce parameters:

```python
# Standard convolution (v2)
nn.Conv1d(in_ch, out_ch, kernel_size=7)  # in_ch × out_ch × 7 parameters

# Depthwise separable (v3/v4/v5)
nn.Conv1d(in_ch, in_ch, kernel_size=3, groups=in_ch)  # in_ch × 3
+ nn.Conv1d(in_ch, out_ch, kernel_size=1)             # in_ch × out_ch
# Total: in_ch × 3 + in_ch × out_ch parameters
```

**Benefits:**
- ~5-10× fewer parameters for equivalent channel counts
- Faster inference on CPUs and mobile accelerators
- Better for knowledge distillation (less capacity to overfit)

---

## Model Selection Guide

### By Use Case

| Use Case | Recommended Model | Rationale |
|----------|------------------|-----------|
| **Research baseline** | v2 | Most parameters, best potential accuracy |
| **Development/prototyping** | v3 | Balanced, easy to train |
| **Real-time CPU** | v4 | Fast inference, good accuracy |
| **Raspberry Pi** | v5 | Smallest, quantization-ready |
| **Embedded systems** | v5 | Minimal memory, INT8 support |
| **Cloud inference** | v2 or v3 | Can afford larger models |

### By Hardware Constraints

| Device | CPU | RAM | Recommended Model |
|--------|-----|-----|-------------------|
| **Workstation/Server** | High | Abundant | v2 or v3 |
| **Laptop** | Medium | 8-16 GB | v3 or v4 |
| **Raspberry Pi 5** | 2.4 GHz ARM | 4-8 GB | v4 or v5 |
| **Raspberry Pi 4** | 1.5 GHz ARM | 2-4 GB | v5 (quantized) |
| **Jetson Nano** | GPU | 4 GB | v4 or v5 |
| **Microcontroller** | <1 GHz | <1 GB | v5 (INT8) |

---

## Quantization Example (v5 Only)

```python
import torch
from models.xn_xiao_net_v5 import XiaoNetEdge

# Load trained model
model = XiaoNetEdge(base_channels=4)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Prepare for quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibrate with representative data
with torch.no_grad():
    for data in calibration_loader:
        model(data)

# Convert to INT8
model_int8 = torch.quantization.convert(model, inplace=False)

# Model size reduced ~4×, inference 2-4× faster on CPU
torch.save(model_int8.state_dict(), 'xiaonet_edge_int8.pth')
```

---

## Common Parameters

All models share these initialization parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_len` | int | 1000 | Input sequence length (samples) |
| `in_channels` | int | 3 | Input channels (Z, N, E components) |
| `num_phases` | int | 3 | Output classes (P, S, Noise) |
| `base_channels` | int | varies | Controls model width |

---

## References

- **MobileNetV2:** Sandler et al. (2018) - Depthwise separable convolutions
- **EfficientNet:** Tan & Le (2019) - Hardswish activation, scaling strategies
- **U-Net:** Ronneberger et al. (2015) - Encoder-decoder with skip connections
- **PhaseNet:** Zhu & Beroza (2019) - Seismic phase picking architecture

---

## Next Steps

- [Deployment Guide](deployment.md) - How to deploy each model variant
- [Training Guide](training.md) - How to train and fine-tune models
- [Benchmarks](benchmarks.md) - Performance comparisons across hardware
