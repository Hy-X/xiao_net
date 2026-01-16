# XiaoNet: Lightweight Neural Network for Seismic Phase Picking and Earthquake Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**XiaoNet** is a lightweight, edge-oriented neural network framework for **seismic phase picking**, **earthquake detection**, and **seismological signal processing**. Designed as a compact and efficient alternative to large deep-learning models such as PhaseNet and EQTransformer, this PyTorch-based framework focuses on **model compression**, **knowledge distillation**, **edge AI**, and **streaming-friendly inference** for low-power devices (e.g., Raspberry Pi 5, embedded systems, IoT devices).

## Keywords

`seismic phase picking` | `earthquake detection` | `neural network` | `deep learning` | `edge AI` | `model compression` | `knowledge distillation` | `PyTorch` | `seismology` | `U-Net` | `PhaseNet` | `EQTransformer` | `STA/LTA` | `real-time inference` | `edge computing` | `seismic waveform processing` | `earthquake early warning` | `seismological machine learning`

---

## Motivation

Traditional deep-learning-based seismic phase pickers (such as PhaseNet, EQTransformer, and GPD) provide excellent accuracy for earthquake detection and phase picking but are often too heavy for real-time, low-power, or embedded systems. As a result, many operational edge deployments, seismic monitoring networks, and earthquake early warning systems still rely on classical methods such as STA/LTA (Short-Term Average/Long-Term Average).

**XiaoNet** aims to bridge this gap by:
- Preserving the accuracy benefits of modern deep learning and neural networks
- Drastically reducing model size and inference latency for edge devices
- Supporting continuous waveform processing and real-time seismic monitoring
- Enabling deployment on resource-constrained hardware (Raspberry Pi, microcontrollers, edge servers)
- Providing an open-source alternative for seismological research and operational systems

---

## Key Features

- 🪶 **Lightweight U-Net-style architectures** for seismic phase picking and earthquake detection
- 🎓 **Knowledge distillation** from large pretrained teacher models (e.g., PhaseNet, EQTransformer)
- ⚡ **Edge-first design**: low memory footprint, minimal compute requirements, ultra-low latency inference
- 🔁 **Streaming-friendly inference** on continuous seismic waveforms and real-time data streams
- 🧩 **Modular codebase** for reusability, experimentation, and easy integration
- 📦 **Config-driven experiments** using JSON configuration files
- 🌐 **PyTorch-based** implementation for easy deployment and model export
- 📊 **Compatible with SeisBench** and standard seismological data formats (SAC, MiniSEED, etc.)
- 🔬 **Research-ready** with comprehensive evaluation metrics and benchmarking tools

---

## Project Structure

```
xiao_net/
├── README.md
├── LICENSE
├── requirements.txt            # Python package dependencies
├── config.json                 # Default hyperparameters & paths
├── xn_main_train.py            # Main training script
├── xn_utils.py                 # Utility functions (seed, device, helpers)
├── xn_early_stopping.py        # Early stopping class
│
├── models/
│   ├── __init__.py
│   └── xn_xiao_net.py          # XiaoNet student network architecture
│
├── loss/
│   ├── __init__.py
│   └── xn_distillation_loss.py # Distillation + label loss
│
├── dataloader/
│   ├── __init__.py
│   └── xn_loaders.py           # DataLoaders wrapping GenericGenerator
│
├── augmentations/
│   ├── __init__.py
│   └── xn_augment_pipeline.py  # WindowAroundSample, Normalize, etc.
│
├── evaluation/
│   ├── __init__.py
│   └── xn_evaluate.py          # Evaluation functions and metrics
│
├── experiments/                # Example experiment scripts
│   └── xn_demo_10s.py          # Demo experiment for 10-second windows
│
└── tests/                      # Unit tests for modules
    ├── __init__.py
    ├── xn_test_models.py       # Model architecture tests
    └── xn_test_loss.py         # Loss function tests
```

---

## Use Cases and Applications

**XiaoNet** is designed for various seismological and geophysical applications:

- **Earthquake Early Warning (EEW) Systems**: Real-time phase picking for rapid earthquake detection
- **Seismic Monitoring Networks**: Continuous waveform analysis on edge devices
- **Research and Education**: Lightweight alternative for seismological machine learning research
- **Embedded Seismic Stations**: Deployment on Raspberry Pi, microcontrollers, and IoT devices
- **Field Deployments**: Low-power seismic phase picking in remote locations
- **Real-time Seismology**: Streaming inference on continuous seismic data streams
- **Model Compression Research**: Benchmarking knowledge distillation techniques for seismology

---

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 1.9+ (CPU or GPU support)
- CUDA (optional, for GPU acceleration)

### Quick Install

```bash
git clone https://github.com/yourname/xiao_net.git
cd xiao_net
pip install -r requirements.txt
```

### Core Dependencies

- **PyTorch**: Deep learning framework for neural network training and inference
- **NumPy**: Numerical computing and array operations
- **ObsPy**: Seismological data processing and waveform handling
- **SeisBench**: Seismological machine learning models and datasets
- **SciPy**: Scientific computing and signal processing
- **Matplotlib**: Visualization and plotting

### Optional Dependencies

- **tqdm**: Progress bars for training loops

---

## Quick Start

### Train a student model with distillation

```bash
python xn_main_train.py --config config.json
```

### Run a demo experiment

```bash
python experiments/xn_demo_10s.py
```

---

## Related Work and Comparisons

**XiaoNet** is inspired by and designed as a lightweight alternative to:

- **PhaseNet**: Deep learning model for seismic phase picking
- **EQTransformer**: Transformer-based earthquake detection and phase picking
- **GPD (Generalized Phase Detection)**: Machine learning approach to seismic phase detection
- **STA/LTA**: Classical short-term average/long-term average trigger algorithm

**Key Differences:**
- **Model Size**: XiaoNet models are 10-100x smaller than PhaseNet/EQTransformer
- **Inference Speed**: Optimized for real-time processing on edge devices
- **Deployment**: Designed for resource-constrained environments
- **Architecture**: U-Net-based student models trained via knowledge distillation

---

## Design Philosophy

1. **Edge-first constraints**
   - Model size < 1–5 MB
   - Real-time or near-real-time inference
   - Low memory footprint

2. **Modularity over monoliths**
   - Every major function in its own module
   - Clear input/output contracts
   - Easy to swap components

3. **Reproducibility**
   - All hyperparameters stored in `config.json`
   - Deterministic training options
   - Versioned experiments

4. **Backward safety**
   - All historical scripts preserved in `archive/`
   - No breaking changes without versioning

---

## Research Goals

This project explores the following research directions in **seismological machine learning** and **edge AI**:

- **Model Compression**: How small can a neural phase picker be while still outperforming classical methods like STA/LTA?
- **Accuracy-Size Trade-offs**: What is the optimal trade-off between input window length, model size, and picking accuracy?
- **Knowledge Distillation**: How does distillation affect temporal precision of P- and S-wave picks?
- **Architecture Design**: What neural network architectures are most suitable for continuous waveform processing?
- **Edge Performance**: How does power consumption, memory usage, and inference latency scale with model complexity?
- **Real-time Processing**: Can lightweight models achieve real-time seismic phase picking on edge devices?
- **Generalization**: How well do compressed models generalize across different seismic networks and regions?

---

## Example Use Case

### Basic Usage: Seismic Phase Picking

```python
from models.xn_xiao_net import XiaoNet
import torch

# Initialize model for 3-component seismic data (Z, N, E)
model = XiaoNet(window_len=1000, in_channels=3, num_phases=3, base_channels=16)

# Input: batch_size=1, channels=3 (ZNE), samples=1000
x = torch.randn(1, 3, 1000)

# Forward pass: outputs phase probability distributions
y = model(x)  # Shape: (1, 3, 1000) for P, S, and noise probabilities
```

### Training with Knowledge Distillation

```python
from models.xn_xiao_net import XiaoNet
from loss.xn_distillation_loss import DistillationLoss
from xn_utils import set_seed, setup_device
from xn_main_train import main

# Set up training
set_seed(42)
device = setup_device('cuda')

# Initialize model
model = XiaoNet(window_len=1000, in_channels=3, num_phases=3).to(device)

# Initialize distillation loss
criterion = DistillationLoss(alpha=0.5, temperature=4.0)

# Run training (see xn_main_train.py for full implementation)
main('config.json')
```

### Data Augmentation

```python
from augmentations.xn_augment_pipeline import AugmentPipeline, Normalize, AddNoise

# Create augmentation pipeline
augmentations = [
    Normalize(method='zscore'),
    AddNoise(noise_level=0.1)
]
pipeline = AugmentPipeline(augmentations)

# Apply to waveform
augmented_waveform = pipeline(waveform)
```

---

## Performance and Benchmarks

**XiaoNet** is designed to achieve:

- **Model Size**: < 1-5 MB (vs. 50-200 MB for PhaseNet/EQTransformer)
- **Inference Latency**: < 10 ms per window on CPU, < 5 ms on GPU
- **Memory Footprint**: < 100 MB during inference
- **Accuracy**: Comparable to or better than STA/LTA, approaching PhaseNet performance
- **Power Consumption**: Optimized for low-power edge devices

*Note: Specific benchmarks and performance metrics will be published in upcoming research papers.*

---

## Documentation Standards

Each module in this project follows these rules:

- Clear module-level docstring describing purpose
- Explicit input/output descriptions
- Example usage in docstrings
- No hidden side effects
- Minimal hard-coded constants

---

## Roadmap

- [ ] Baseline student U-Net implementation
- [ ] Distillation training pipeline
- [ ] Streaming inference engine
- [ ] Raspberry Pi 5 benchmark
- [ ] Comparison with STA/LTA
- [ ] Paper-ready evaluation suite

---

## Citation

If you use **XiaoNet** in your research on seismic phase picking, earthquake detection, edge AI, or model compression, please cite:

```bibtex
@misc{xiaonet2026,
  title={XiaoNet: Lightweight Neural Phase Picking for Edge Devices},
  author={Xiao, Hongyu},
  year={2026},
  howpublished={\url{https://github.com/yourname/xiao_net}},
  note={Lightweight neural network framework for seismic phase picking and earthquake detection}
}
```

### Related Citations

When using XiaoNet, you may also want to cite:

- **PhaseNet**: Zhu, W., & Beroza, G. C. (2019). PhaseNet: a deep-neural-network-based seismic arrival-time picking method. *Geophysical Journal International*.
- **EQTransformer**: Mousavi, S. M., et al. (2020). Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. *Nature Communications*.
- **SeisBench**: Woollam, J., et al. (2022). SeisBench—A Toolbox for Machine Learning in Seismology. *Seismological Research Letters*.

---

## Contributing

Contributions are welcome! This project is open to:

- Bug reports and feature requests
- Code contributions and improvements
- Documentation enhancements
- Performance optimizations
- Benchmark results and evaluations
- Integration with other seismological tools

Please open an issue or submit a pull request on GitHub.

---

## Acknowledgments

**XiaoNet** builds upon the excellent work of:

- **PhaseNet** (Zhu & Beroza, 2019) for seismic phase picking
- **EQTransformer** (Mousavi et al., 2020) for earthquake detection
- **EasyQuake** for earthquake detection and cataloging workflows
- **SeisBench** for seismological machine learning infrastructure
- **ObsPy** for seismic data processing
- The broader seismological and machine learning communities

---

## License

MIT License

Copyright (c) 2026 Hongyu Xiao

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

---

## Contact

For questions, collaborations, research inquiries, or issues, please:

- **Open an issue** on GitHub for bug reports and feature requests
- **Contact directly** for research collaborations and partnerships

**Hongyu Xiao**  
Email: hongyu.xiao-1@ou.edu  
Affiliation: University of Oklahoma

---

## Additional Resources

- **Seismology**: [IRIS](https://www.iris.edu/), [OGS](https://www.ou.edu/ogs)
- **Machine Learning**: [PyTorch Documentation](https://pytorch.org/docs/), [SeisBench](https://github.com/seisbench/seisbench)
- **Data Sources**: [IRIS DMC](https://ds.iris.edu/ds/), [OGS](https://www.ou.edu/ogs)

