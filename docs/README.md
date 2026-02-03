# XiaoNet Documentation

Welcome to the XiaoNet documentation. This guide provides comprehensive information for training, deploying, and using XiaoNet models for seismic phase picking.

## Quick Navigation

### Getting Started
- [Installation](../README.md#installation) - Set up your environment
- [Quick Start](../README.md#quick-start) - Run your first experiment

### Core Documentation
- **[Model Architecture](model-architecture.md)** - Understanding the five XiaoNet variants (v1-v5)
- **[Data Pipeline](data-pipeline.md)** - Data loading, preprocessing, and augmentation
- **[Training Guide](training.md)** - How to train and fine-tune models *(coming soon)*
- **[Deployment Guide](deployment.md)** - Deploy models to edge devices *(coming soon)*

### Advanced Topics
- **[Evaluation](evaluation.md)** - Metrics and benchmarking *(coming soon)*
- **[Quantization](quantization.md)** - INT8 optimization for edge devices *(coming soon)*
- **[API Reference](api-reference.md)** - Detailed API documentation *(coming soon)*

---

## Documentation Overview

### Model Architecture

Learn about the five XiaoNet variants and their trade-offs:

| Model | Parameters | Best For |
|-------|-----------|----------|
| v2 | ~15K | Baseline accuracy |
| v3 | ~4K | Balanced speed/accuracy |
| v4 | ~3K | Real-time CPU inference |
| v5 | ~1.5K | Edge devices, INT8 quantization |

[Read the Model Architecture Guide →](model-architecture.md)

### Data Pipeline

Understand how to:
- Load SeisBench datasets
- Configure augmentation pipelines
- Handle multiple phase types
- Create PyTorch DataLoaders
- Avoid common pitfalls (variable-length tensors, etc.)

[Read the Data Pipeline Guide →](data-pipeline.md)

---

## Key Concepts

### Seismic Phase Picking

XiaoNet performs **seismic phase picking**: identifying P-wave and S-wave arrival times in continuous seismic waveforms. This is critical for:

- **Earthquake detection and location**
- **Earthquake early warning (EEW)**
- **Real-time seismic monitoring**
- **Seismological research**

### Model Design Philosophy

XiaoNet is designed for **edge deployment**:

1. **Minimal parameters** (1.5K-15K vs 50K-1M for PhaseNet)
2. **Fast inference** (<10ms per window on CPU)
3. **Low memory footprint** (<100MB during inference)
4. **Quantization-ready** (INT8 support in v5)

### Training Strategy

XiaoNet uses **transfer learning**:

1. Start with PhaseNet weights (pretrained on STEAD)
2. Fine-tune on regional datasets (e.g., OKLA_1Mil)
3. Optional: Apply knowledge distillation from teacher model

---

## Common Workflows

### Training a New Model

```bash
# 1. Prepare your config
vim config.json

# 2. Train model
python xn_main_train.py --config config.json

# 3. Evaluate
python evaluation/xn_evaluate.py --model checkpoints/best_model.pth
```

### Deploying to Raspberry Pi

```bash
# 1. Export to ONNX
python scripts/export_onnx.py --model v5 --output xiaonet_edge.onnx

# 2. Quantize to INT8
python scripts/quantize_model.py --input xiaonet_edge.onnx

# 3. Deploy to device
scp xiaonet_edge_int8.onnx pi@raspberrypi:/home/pi/models/
```

### Streaming Inference

```python
from models.xn_xiao_net_v5 import XiaoNetEdge
import torch

# Load model
model = XiaoNetEdge()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Process continuous stream
for window in stream_generator:
    with torch.no_grad():
        probs = model(window)  # (3, 1000) - P, S, Noise
        
    # Detect phases
    p_arrival = detect_peak(probs[0])
    s_arrival = detect_peak(probs[1])
```

---

## Project Structure Reference

```
xiao_net/
├── docs/                           # Documentation (you are here)
│   ├── README.md                   # This file
│   ├── model-architecture.md       # Model variants guide
│   ├── data-pipeline.md            # Data loading guide
│   ├── training.md                 # Training guide (coming soon)
│   └── deployment.md               # Deployment guide (coming soon)
│
├── models/                         # Model architectures
│   ├── xn_xiao_net_v2.py          # Speed-optimized baseline
│   ├── xn_xiao_net_v3.py          # Balanced variant
│   ├── xn_xiao_net_v4.py          # CPU-optimized
│   └── xn_xiao_net_v5.py          # Edge-optimized
│
├── dataloader/                     # Data loading utilities
│   └── xn_loaders.py              # PyTorch DataLoader wrappers
│
├── augmentations/                  # Data augmentation
│   └── xn_augment_pipeline.py     # Augmentation pipeline
│
├── loss/                           # Loss functions
│   └── xn_distillation_loss.py    # Distillation + label loss
│
├── evaluation/                     # Evaluation tools
│   └── xn_evaluate.py             # Metrics and benchmarks
│
├── experiments/                    # Example scripts
│   └── xn_demo_10s.py             # Demo experiment
│
├── tests/                          # Unit tests
│   ├── xn_test_models.py          # Model tests
│   └── xn_test_loss.py            # Loss function tests
│
├── xn_main_train.py               # Main training script
├── xn_utils.py                    # Utility functions
├── xn_early_stopping.py           # Early stopping
├── config.json                     # Default configuration
└── requirements.txt                # Dependencies
```

---

## FAQ

### Which model should I use?

- **Prototyping/development**: v3 (balanced)
- **Real-time CPU**: v4
- **Raspberry Pi / embedded**: v5
- **Research baseline**: v2

See the [Model Architecture Guide](model-architecture.md#model-selection-guide) for details.

### How do I handle variable-length tensors?

Always use `strategy="pad"` in windowing operations:

```python
sbg.WindowAroundSample(..., strategy="pad")  # ✅ Correct
sbg.WindowAroundSample(..., strategy="variable")  # ❌ Causes errors
```

See [Data Pipeline - Common Pitfalls](data-pipeline.md#common-pitfalls).

### Can XiaoNet run in real-time?

Yes! XiaoNet v4 and v5 are designed for real-time inference:

- **v4**: ~5-10ms per window on modern CPU
- **v5**: ~3-5ms per window, <2ms with INT8 quantization

### What sampling rate is required?

XiaoNet is typically trained on **100 Hz** data, but can be adapted to other rates:

- **100 Hz**: Standard for most applications
- **50 Hz**: Acceptable with adjusted window lengths
- **200+ Hz**: Consider downsampling to reduce compute

### How do I add custom phase types?

Update the phase dictionary in your augmentation pipeline:

```python
phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_custom_phase_sample": "P",  # Add your phase
    # ... more phases
}
```

See [Data Pipeline - Phase Type Mapping](data-pipeline.md#phase-type-mapping).

---

## Contributing to Documentation

Documentation improvements are welcome! To contribute:

1. **Fork the repository**
2. **Edit markdown files** in `docs/`
3. **Test with markdownlint**: `markdownlint docs/*.md`
4. **Submit a pull request**

### Documentation Standards

- Use clear, concise language for developer audience
- Include code examples for concepts
- Link related sections with relative links
- Keep examples practical and runnable
- Add tables for comparisons
- Use callouts for important notes (⚠️, ✅, ❌)

---

## External Resources

### Seismology & Machine Learning

- [SeisBench](https://github.com/seisbench/seisbench) - Seismological ML toolkit
- [PhaseNet Paper](https://doi.org/10.1093/gji/ggy423) - Original phase picking architecture
- [ObsPy](https://docs.obspy.org/) - Seismic data processing

### Deep Learning & Optimization

- [PyTorch Documentation](https://pytorch.org/docs/)
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform inference
- [TensorFlow Lite](https://www.tensorflow.org/lite) - Mobile/edge deployment

### Hardware Platforms

- [Raspberry Pi](https://www.raspberrypi.org/) - Popular edge platform
- [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) - GPU-accelerated edge
- [ARM NN](https://www.arm.com/products/silicon-ip-cpu/ethos/arm-nn) - NPU optimization

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourname/xiao_net/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourname/xiao_net/discussions)
- **Email**: hongyu.xiao-1@ou.edu

---

## License

This documentation is part of the XiaoNet project and is licensed under the MIT License.

Copyright (c) 2026 Hongyu Xiao
