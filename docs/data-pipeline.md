# Data Pipeline and Augmentation

This document describes the data loading, preprocessing, and augmentation pipeline for training XiaoNet models.

## Overview

XiaoNet uses [SeisBench](https://github.com/seisbench/seisbench) for data management and augmentation. The pipeline consists of:

1. **Dataset loading** from SeisBench or custom sources
2. **Data augmentation** with windowing, normalization, and probabilistic labeling
3. **PyTorch DataLoaders** for batch processing during training

---

## Dataset Support

### SeisBench Datasets

XiaoNet is compatible with standard SeisBench datasets:

- **STEAD** (Seismic Waveform Dataset)
- **INSTANCE** (Italian Seismic Dataset)
- **ETHZ** (Swiss Seismological Service)
- Custom regional datasets (e.g., OKLA_1Mil)

### Loading Data

```python
import seisbench.data as sbd

# Load STEAD dataset
data = sbd.STEAD()

# Split into train/dev/test
train = data.train()
dev = data.dev()
test = data.test()
```

### Custom Dataset Integration

For custom seismic datasets:

```python
# Create a SeisBench-compatible dataset
from seisbench.data import WaveformDataset

custom_data = WaveformDataset(
    path="path/to/dataset",
    sampling_rate=100,
    component_order="ZNE"
)
```

---

## Data Augmentation Pipeline

### Standard Pipeline (Training)

```python
import seisbench.generate as sbg
import numpy as np

# Define phase type mapping
phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}

# Create augmentation pipeline
augmentations = [
    # 1. Extract window around phase arrival
    sbg.WindowAroundSample(
        list(phase_dict.keys()),
        samples_before=3000,    # 30 seconds before at 100 Hz
        windowlen=6000,         # 60 seconds total
        selection="random",
        strategy="pad"          # Pad if insufficient data
    ),
    
    # 2. Extract fixed window for consistent input size
    sbg.FixedWindow(
        p0=3000-1500,          # Center around phase arrival
        windowlen=3001,        # 30.01 seconds at 100 Hz
        strategy="pad"
    ),
    
    # 3. Normalize waveform
    sbg.Normalize(
        demean_axis=-1,        # Remove mean
        detrend_axis=-1,       # Remove linear trend
        amp_norm_axis=-1,      # Normalize amplitude
        amp_norm_type="peak"   # Peak normalization
    ),
    
    # 4. Convert to float32
    sbg.ChangeDtype(np.float32),
    
    # 5. Create probabilistic labels
    sbg.ProbabilisticLabeller(
        sigma=30,              # Gaussian width (0.3 seconds at 100 Hz)
        dim=0                  # Apply along first dimension
    ),
]

# Apply to data generator
train_generator = sbg.GenericGenerator(train)
train_generator.add_augmentations(augmentations)
```

### Validation/Test Pipeline (No Augmentation)

```python
# Simpler pipeline for evaluation
val_augmentations = [
    sbg.WindowAroundSample(
        list(phase_dict.keys()),
        samples_before=3000,
        windowlen=6000,
        selection="first",      # Consistent window selection
        strategy="pad"
    ),
    sbg.FixedWindow(p0=3000-1500, windowlen=3001, strategy="pad"),
    sbg.Normalize(demean_axis=-1, detrend_axis=-1, 
                  amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(sigma=30, dim=0),
]

val_generator = sbg.GenericGenerator(dev)
val_generator.add_augmentations(val_augmentations)
```

---

## Augmentation Components

### 1. WindowAroundSample

Extracts a window centered around a seismic phase arrival.

**Parameters:**
- `keys`: List of phase arrival column names
- `samples_before`: Samples before phase arrival to include
- `windowlen`: Total window length
- `selection`: How to select phase if multiple present (`"random"`, `"first"`, `"last"`)
- `strategy`: How to handle edge cases (`"pad"`, `"variable"`)

**Common Issue:** Using `strategy="variable"` causes variable-length outputs that cannot be batched. **Always use `strategy="pad"`** for consistent tensor shapes.

### 2. FixedWindow / RandomWindow

Extracts a fixed-length window from the waveform.

**FixedWindow:**
```python
sbg.FixedWindow(
    p0=1500,           # Start position
    windowlen=3001,    # Window length
    strategy="pad"     # Pad if needed
)
```

**RandomWindow:**
```python
sbg.RandomWindow(
    windowlen=3001,    # Window length
    strategy="pad"     # Pad if needed
)
```

⚠️ **Important:** Random windows should have fixed length (`strategy="pad"`) to ensure batchability.

### 3. Normalize

Applies waveform normalization for training stability.

**Types:**
- **Demean** (`demean_axis=-1`): Remove mean value
- **Detrend** (`detrend_axis=-1`): Remove linear trend
- **Amplitude normalization** (`amp_norm_axis=-1`):
  - `amp_norm_type="peak"`: Divide by maximum absolute value
  - `amp_norm_type="std"`: Divide by standard deviation

```python
sbg.Normalize(
    demean_axis=-1,
    detrend_axis=-1,
    amp_norm_axis=-1,
    amp_norm_type="peak"
)
```

### 4. ProbabilisticLabeller

Converts discrete phase arrivals into Gaussian probability distributions.

**Why:** Helps model learn smooth probability distributions rather than hard one-hot labels.

```python
sbg.ProbabilisticLabeller(
    sigma=30,          # Standard deviation (samples)
    dim=0              # Dimension to apply labeling
)
```

**Effect:**
- P-wave arrival at sample 1000 → Gaussian peak at 1000 with σ=30 samples (0.3s at 100 Hz)
- Creates overlapping distributions if P and S waves are close
- Background noise has uniform low probability

---

## Phase Type Mapping

XiaoNet supports multiple regional phase types mapped to P and S classes:

| Original Phase | Mapped Class | Description |
|---------------|--------------|-------------|
| `trace_p_arrival_sample` | P | First-arriving P-wave |
| `trace_P_arrival_sample` | P | Standard P-wave |
| `trace_Pg_arrival_sample` | P | Regional crustal P-wave |
| `trace_Pn_arrival_sample` | P | Moho-refracted P-wave |
| `trace_PmP_arrival_sample` | P | Moho-reflected P-wave |
| `trace_s_arrival_sample` | S | First-arriving S-wave |
| `trace_S_arrival_sample` | S | Standard S-wave |
| `trace_Sg_arrival_sample` | S | Regional crustal S-wave |
| `trace_Sn_arrival_sample` | S | Moho-refracted S-wave |
| `trace_SmS_arrival_sample` | S | Moho-reflected S-wave |

---

## DataLoader Configuration

### Creating DataLoaders

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_generator,
    batch_size=32,
    num_workers=4,        # Parallel data loading
    shuffle=True,         # Shuffle training data
    pin_memory=True,      # Faster GPU transfer
    drop_last=True        # Drop incomplete batches
)

val_loader = DataLoader(
    val_generator,
    batch_size=64,        # Can use larger batches for validation
    num_workers=4,
    shuffle=False,        # No shuffling for validation
    pin_memory=True
)
```

### Batch Structure

Each batch contains:

```python
for batch in train_loader:
    X = batch["X"]        # (batch_size, 3, 3001) - Waveforms
    y = batch["y"]        # (batch_size, 3, 3001) - Phase probabilities
    metadata = batch.get("metadata", {})  # Optional metadata
```

**Tensor Shapes:**
- `X`: `(B, C, T)` where B=batch, C=3 channels (Z,N,E), T=time samples
- `y`: `(B, P, T)` where P=3 phases (P, S, Noise)

---

## Common Pitfalls

### ❌ Variable-Length Tensors

**Problem:**
```python
sbg.WindowAroundSample(..., strategy="variable")
# Creates tensors of different lengths → DataLoader error
```

**Error:**
```
RuntimeError: stack expects each tensor to be equal size, 
but got [3, 12001] at entry 0 and [3, 12002] at entry 2
```

**Solution:**
```python
sbg.WindowAroundSample(..., strategy="pad")
# Always pads to consistent length
```

### ❌ Missing Normalization

**Problem:**
```python
# No normalization
augmentations = [sbg.WindowAroundSample(...)]
```

**Effect:** Raw seismic amplitudes vary widely → unstable training

**Solution:**
```python
augmentations = [
    sbg.WindowAroundSample(...),
    sbg.Normalize(demean_axis=-1, detrend_axis=-1, 
                  amp_norm_axis=-1, amp_norm_type="peak"),
]
```

### ❌ Wrong Data Type

**Problem:**
```python
# Default dtype may be float64
```

**Effect:** 2× memory usage, slower training

**Solution:**
```python
augmentations.append(sbg.ChangeDtype(np.float32))
```

---

## Advanced Augmentations

### Noise Injection

```python
sbg.AddGaussianNoise(scale=0.1)  # Add random noise
```

### Amplitude Scaling

```python
sbg.RandomScaling(scale_range=(0.5, 2.0))  # Random amplitude scaling
```

### Time Shift

```python
sbg.RandomShift(max_shift=100)  # Shift waveform up to 1 second
```

### Example Advanced Pipeline

```python
advanced_augmentations = [
    sbg.WindowAroundSample(list(phase_dict.keys()), 
                          samples_before=3000, windowlen=6000,
                          selection="random", strategy="pad"),
    sbg.FixedWindow(p0=1500, windowlen=3001, strategy="pad"),
    
    # Advanced augmentations
    sbg.RandomScaling(scale_range=(0.8, 1.2)),     # ±20% amplitude
    sbg.AddGaussianNoise(scale=0.05),              # 5% noise
    
    # Standard normalization
    sbg.Normalize(demean_axis=-1, detrend_axis=-1,
                  amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(sigma=30, dim=0),
]
```

---

## Sampling Strategy

### Balanced Sampling

For datasets with class imbalance (many noise windows, few events):

```python
from seisbench.generate import SamplingStrategy

# Sample equal numbers from event/noise
sampler = SamplingStrategy(
    strategy="balanced",
    classes=["event", "noise"],
    weights=[0.5, 0.5]
)

train_generator.add_sampler(sampler)
```

### Magnitude-Based Sampling

```python
# Oversample larger earthquakes
sampler = SamplingStrategy(
    strategy="magnitude_weighted",
    min_magnitude=1.0,
    max_magnitude=7.0,
    weight_function=lambda m: np.exp(m)  # Exponential weighting
)
```

---

## Data Pipeline Checklist

Before training, verify:

- ✅ All windows have **consistent length** (use `strategy="pad"`)
- ✅ **Normalization** is applied (demean, detrend, amp_norm)
- ✅ **Data type** is float32 (not float64)
- ✅ **Probabilistic labels** have appropriate σ (typically 20-50 samples)
- ✅ **Phase dictionary** includes all relevant phase types
- ✅ **Batch size** fits in memory
- ✅ **num_workers** is set (typically 4-8 for faster loading)

---

## References

- [SeisBench Documentation](https://seisbench.readthedocs.io/)
- [SeisBench Paper](https://pubs.geoscienceworld.org/ssa/srl/article/93/3/1695/612005)
- [PhaseNet Data Augmentation](https://github.com/wayneweiqiang/PhaseNet)

---

## Next Steps

- [Training Guide](training.md) - How to train models with this pipeline
- [Model Architecture](model-architecture.md) - Understanding XiaoNet models
- [Evaluation](evaluation.md) - How to evaluate trained models
