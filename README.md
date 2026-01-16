# xiao_net


**xiao_net** is a lightweight, edge-oriented neural network framework for seismic phase picking, designed as a compact and efficient alternative to large deep-learning models such as PhaseNet and EQTransformer. The project focuses on **model compression**, **knowledge distillation**, and **streaming-friendly inference** for low-power devices (e.g., Raspberry Pi 5).

---

## Motivation

Traditional deep-learning-based seismic pickers provide excellent accuracy but are often too heavy for real-time, low-power, or embedded systems. As a result, many operational edge deployments still rely on classical methods such as STA/LTA.

**xiao_net** aims to bridge this gap by:
- Preserving the accuracy benefits of modern deep learning
- Drastically reducing model size and inference latency
- Supporting continuous waveform processing
- Enabling deployment on resource-constrained hardware

---

## Key Features

- 🪶 **Lightweight U-Net-style architectures** for seismic phase picking
- 🎓 **Knowledge distillation** from large pretrained teacher models (e.g., PhaseNet)
- ⚡ **Edge-first design**: low memory, low compute, low latency
- 🔁 **Streaming-friendly inference** on continuous waveforms
- 🧩 **Modular codebase** for reusability and experimentation
- 📦 **Config-driven experiments** using `config.json`

---

## Project Structure

```
xiao_net/
├── README.md
├── config.json
├── main.py                     # Entry point for experiments
│
├── models/
│   ├── small_phasenet.py       # Student network definitions
│   ├── teacher_wrappers.py     # Interfaces for teacher models
│   └── __init__.py
│
├── distillation/
│   ├── losses.py               # Distillation losses
│   ├── trainer.py              # Distillation training loops
│   └── __init__.py
│
├── data/
│   ├── loaders.py              # Dataset and DataLoader utilities
│   ├── augmentations.py        # Data augmentation pipelines
│   └── __init__.py
│
├── inference/
│   ├── streaming.py            # Streaming / sliding-window inference
│   ├── postprocess.py          # Peak picking & smoothing
│   └── __init__.py
│
├── utils/
│   ├── early_stopping.py
│   ├── metrics.py
│   ├── logging.py
│   └── __init__.py
│
└── archive/                    # Historical and reference implementations
    ├── original_training.py
    ├── baseline_experiments/
    └── notes.md
```

---

## Installation

```bash
git clone https://github.com/yourname/xiao_net.git
cd xiao_net
pip install -r requirements.txt
```

Dependencies typically include:
- PyTorch
- NumPy
- ObsPy
- SeisBench
- SciPy
- Matplotlib

---

## Quick Start

### Train a student model with distillation

```bash
python main.py --config config.json
```

### Run inference on continuous waveform

```bash
python inference/streaming.py --config config.json --input data/example.mseed
```

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

This project explores the following research directions:

- How small can a neural picker be while still outperforming STA/LTA?
- What is the optimal trade-off between input window length and accuracy?
- How does distillation affect temporal precision of picks?
- What architectures are most suitable for continuous waveform processing?
- How does power consumption scale with model complexity?

---

## Example Use Case

```python
from models.small_phasenet import SmallPhaseNet
import torch

model = SmallPhaseNet(window_len=1000, in_channels=3, num_phases=3)
x = torch.randn(1, 3, 1000)
y = model(x)
```

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

If you use this project in your research, please cite:

```
@misc{xiao_net,
  title={xiao_net: Lightweight Neural Phase Picking for Edge Devices},
  author={Hongyu Xiao},
  year={2026}
}
```

---

## License

MIT License

---

## Contact

For questions, collaborations, or issues, please open an issue or contact:

**Hongyu Xiao**  
Email: hongyu.xiao-1@ou.edu

