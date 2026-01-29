# XiaoNet Project Design Rules

## Project Overview
**XiaoNet** is a lightweight student neural network for seismic phase picking trained via knowledge distillation from PhaseNet (teacher). The goal is to create a compact (~168K parameters, 93% reduction vs PhaseNet) model suitable for edge deployment while maintaining accuracy through teacher guidance.

---

## ⚡ Quick Reference (TL;DR)

### Critical Rules (NEVER VIOLATE)
```python
# Input/Output
input_shape = (batch, 3, 3001)   # E/N/Z channels, 3001 samples
output_shape = (batch, 3, 3001)  # P/S/Noise, must match input

# Model Architecture
kernel_size = 3                   # All Conv1d layers
base_channels = 16                # Controls model size
encoder_levels = 3                # MaxPool downsampling
decoder_levels = 3                # Upsample + skip connections

# Teacher Model (PhaseNet)
teacher_model.eval()              # Always eval mode
teacher.requires_grad = False     # Always frozen
with torch.no_grad():             # Always in no_grad context
    teacher_pred = teacher(X)

# Student Model (XiaoNet)
student_model.train()             # Training mode during epochs
optimizer = Adam(student_model.parameters())  # Student ONLY, never teacher

# Loss Functions
task_loss = CE(student_pred, y_true)          # Ground truth supervision
distill_loss = KL(teacher_soft, student_soft) # Teacher knowledge transfer
total_loss = 0.5 * distill_loss + 0.5 * task_loss  # Combined (alpha=0.5)

# Hyperparameters (default)
temperature = 3.0    # Softens probability distributions
alpha = 0.5          # Balance: teacher (0.5) vs ground truth (0.5)
learning_rate = 1e-3 # Adam optimizer
batch_size = 32-64   # Depends on GPU memory
```

### Pre-Training Validation
```python
# Run before training to verify setup
assert all(not p.requires_grad for p in teacher_model.parameters())
assert any(p.requires_grad for p in student_model.parameters())
assert student_model(torch.randn(2,3,3001).to(device)).shape == (2,3,3001)
```

### Troubleshooting Fast Guide
- **Loss not decreasing?** → Check LR (try 1e-4), verify teacher frozen
- **Dimension error?** → Verify input is 3001, XiaoNet trims to 3000 then pads to 3001
- **Task loss high, train loss low?** → Student over-relies on teacher, reduce alpha to 0.3
- **Both losses plateau?** → LR too low (increase to 1e-3) or model too small (increase base_channels)

---

## Architecture Constraints

### Model Specification
- **Type**: U-Net style encoder-decoder with skip connections
- **Encoder**: 3 levels of downsampling (MaxPool1d with factor 2)
- **Bottleneck**: 1 level between encoder and decoder
- **Decoder**: 3 levels of upsampling (Upsample with factor 2 + skip connections)
- **Kernel Size**: Always 3 for all Conv1d layers (no exceptions)
- **Base Channels**: 16 (controls model size; scales all layer widths)
- **Activation**: ReLU for intermediate layers, Softmax on output

### Input/Output Specification
- **Input Window Length**: 3001 samples (fixed)
- **Input Channels**: 3 (E, N, Z seismic components)
- **Output Channels**: 3 (P phase, S phase, noise)
- **Output Shape**: Must match input shape (batch, 3, 3001)
- **Internal Trimming**: Input internally trimmed to 3000 (multiple of 8) for stable pooling/upsampling, output padded back to 3001

### Critical Dimension Flow
```
Input:  (batch, 3, 3001)
  ↓ trim to 3000
Enc1:   (batch, base_ch, 3000)
Enc2:   (batch, base_ch*2, 1500)  [after pool]
Enc3:   (batch, base_ch*4, 750)   [after pool]
Bottle: (batch, base_ch*8, 375)   [after pool]
Dec3:   (batch, base_ch*4, 750)   [upsample + skip from enc3]
Dec2:   (batch, base_ch*2, 1500)  [upsample + skip from enc2]
Dec1:   (batch, base_ch, 3000)    [upsample + skip from enc1]
Output: (batch, 3, 3001)          [pad back to match input]
```

**Rule**: If input size changes, verify dimension consistency. Use multiple-of-8 trimming for all odd-sized inputs.

---

## Data Pipeline Requirements

### Dataset Configuration
- **Source**: OKLA_1Mil_120s_Ver_3 (SeisbenCH)
- **Sampling Rate**: 100 Hz
- **Component Order**: ENZ (East, North, Vertical)
- **Split Ratio**: train/dev/test via `data.train_dev_test()`

### Augmentation Pipeline (Mandatory Order)
1. **WindowAroundSample**: Extract 6000 samples around phase arrivals, random selection
2. **RandomWindow**: Extract 3001 sample window with padding strategy
3. **Normalize**: Demean, detrend, peak amplitude normalization
4. **ChangeDtype**: Convert to float32
5. **ProbabilisticLabeller**: Generate soft labels with sigma=30

**Rule**: This order is fixed. Labels must be soft (probabilistic), not hard (one-hot).

### Data Loading
- **Batch Size**: From config.json (typically 32-64)
- **Workers**: From config.json, use `worker_seeding` function
- **Pin Memory**: True for GPU training
- **Prefetch Factor**: 4
- **Persistent Workers**: True

---

## Knowledge Distillation Setup

### Model Roles
- **Teacher**: PhaseNet from `seisbench.models` (STEAD pretrained)
  - Status: FROZEN (all parameters `requires_grad=False`)
  - Mode: Always evaluation mode during training
  - Input: Same as student (3001 samples)
  - Output: 3D probability tensor (batch, 3, 3001)
  
- **Student**: XiaoNet (trainable)
  - Status: TRAINABLE (all parameters `requires_grad=True`)
  - Mode: Training mode during epochs, evaluation mode during validation
  - Input: Same as teacher (3001 samples)
  - Output: 3D probability tensor (batch, 3, 3001)

**Rule**: Never unfreeze teacher. Never train teacher parameters. Teacher only in no_grad context.

### Loss Functions

#### Task Loss (Ground Truth Supervision)
```python
task_loss = -(y_true * log(student_pred + eps)).mean(-1).sum(-1).mean()
```
- **What**: Cross-entropy between student predictions and ground truth labels
- **Why**: Provides direct supervision - ensures student learns actual correct answers, not just mimicking teacher
- **When to monitor**: If task_loss stays high (>0.5) after 10 epochs, student isn't learning ground truth properly
- **Interpretation**: Lower = better ground truth accuracy

#### Distillation Loss (Teacher Guidance)
```python
student_soft = student_pred ^ (1/T)  # Temperature scaling
teacher_soft = teacher_pred ^ (1/T)
distill_loss = KL(teacher_soft || student_soft) * T^2
```
- KL divergence between softened distributions
- Temperature T=3.0 (must be consistent across training)
- Scaled by T² to  (Training Loss)
```python
total_loss = alpha * distill_loss + (1 - alpha) * task_loss
```
- **What**: The actual loss used for backpropagation during training
- **Components**:
  - `alpha * distill_loss`: Weight given to teacher knowledge (soft targets)
  - `(1-alpha) * task_loss`: Weight given to ground truth (hard labels)
- **Default alpha=0.5**: Equal balance between teacher and ground truth
- **Adjustment rules**:
  - If `task_loss >> val_loss`: Student over-relies on teacher → **Decrease alpha to 0.3**
  - If `val_loss >> task_loss`: Student ignores teacher → **Increase alpha to 0.7**
- **Why combined**: Pure task loss = standard training (no distillation benefit). Pure distill loss = student becomes copy of teacher (no ground truth grounding). Combination gets best of both.

**Rule**: Temperature and alpha are hyperparameters. Log any changes in experiment notes with justificationgher alpha)

**Rule**: Temperature and alpha are hyperparameters. Log any changes in experiment notes.

---

## Training Loop Rules

### Optimizer Configuration
- **Type**: Adam
- **Learning Rate**: From config.json (typically 1e-3)
- **Scope**: Student model parameters ONLY
- **Never**: Include teacher parameters in optimizer

### Scheduler & Early Stopping
- **Scheduler**: ReduceLROnPlateau
  - Mode: min (validation loss)
  - Factor: 0.5
  - Patience: 3 epochs
  - Verbose: True
  
- **Early Stopping**: Custom EarlyStopping class
  - Patience: From config.json (typically 7 epochs)
  - Saves: checkpoint.pt (best validation), best_model.pth, final_model.pth
  - Verbose: True

### Training Function Signature
```python
train_one_epoch_distillation(
    train_loader, student_model, teacher_model, 
    optimizer, device, temperature=3.0, alpha=0.5
)
```
Returns: (avg_loss, avg_task_loss)

### Validation Function Signature
### Pre-Training Validation Checklist
Before running `train_model_distillation()`, verify:
```python
# 1. Teacher is frozen
assert all(not p.requires_grad for p in teacher_model.parameters()), "Teacher must be frozen!"

# 2. Student is trainable  
assert any(p.requires_grad for p in student_model.parameters()), "Student must be trainable!"

# 3. Optimizer excludes teacher
teacher_params = set(teacher_model.parameters())
optimizer_params = set(optimizer.param_groups[0]['params'])
assert len(teacher_params & optimizer_params) == 0, "Optimizer must NOT include teacher!"

# 4. Output shapes match
test_input = torch.randn(2, 3, 3001).to(device)
student_out = student_model(test_input)
teacher_out = teacher_model(test_input)
assert student_out.shape == teacher_out.shape == (2, 3, 3001), "Output shape mismatch!"

print("✓ All pre-training validations passed!")
```

```python
evaluate_student_model(val_loader, student_model, device)
```
Returns: avg_val_loss (student only, no teacher)

**Rule**: Teacher always in eval mode. Student trains with teacher guidance, evaluates independently.

---

## File Structure Rules

### Core Model File
- **Location**: `/models/xn_xiao_net.py`
- **Class**: `XiaoNet(nn.Module)`
- **Constructor Args**: window_len, in_channels, num_phases, base_channels
- **Forward Input**: (batch, channels, samples)
- **Forward Output**: (batch, num_phases, samples) with softmax activation
- **Must Have**: Internal trimming to multiple of 8, output padding to original size

### Configuration File
- **Location**: `/config.json` (root)
- **Required Keys**:
  - `device`: {use_cuda, device_id}
  - `training`: {batch_size, num_workers, learning_rate, epochs, patience}
  - `peak_detection`: {sampling_rate, height, distance}

### Training Notebook
- **Location**: `/archive/TL_PNet_1Mil_ModelTrain.ipynb`
- **Cell Order**: (mandatory sequence)
  1. Imports
  2. Config load
  3. Random seeds
  4. PhaseNet load
  5. Device setup
  6. PhaseNet info
  7. XiaoNet import & create
  8. XiaoNet reload (with dimension test)
  9. Data load
  10. Data split
  11. Augmentations
  12. Data loaders
  13. Loss functions (teacher + distillation)
  14. Teacher-Student config
  15. Optimizer
  16. Training functions
  17. EarlyStopping class
  18. Scheduler & training loop
  19. Training execution

---

## Hyperparameter Guidelines

### Fixed Hyperparameters (Do Not Change)
- Window length: 3001
- Kernel size: 3
- Input channels: 3
- Output channels: 3
- Encoder/Decoder levels: 3 each
- Distillation temperature: 3.0
- Alpha (task/distill weight): 0.5
- Scheduler factor: 0.5
- Scheduler patience: 3

### Tunable Hyperparameters (Experiment-Safe)
- Base channels: 16 (can increase for larger model: 24, 32)
- Batch size: 32-128 (depends on GPU memory)
- Learning rate: 1e-3 to 1e-4 (fine-tune if needed)
- EarlyStopping patience: 5-10 epochs
- Temperature (distillation): 2.0-5.0 (higher = softer targets)
- Alpha: 0.3-0.7 (0.5 = equal weight)

**Rule**: Always log any hyperparameter changes in experiment notes with justification.

---

## Post-Training Pipeline

### Model Saving
1. **Best Model**: Loaded from checkpoint.pt at end of training
2. **FP32 Model**: Default trained state (32-bit floats)
3. **INT8 Quantization**: Post-training quantization for edge deployment

### Quantization Rules
- **Type**: torch.quantization.quantize_dynamic()
- **Layers**: Conv1d, Linear
- **Dtype**: qint8
- **Size Reduction**: ~4x (FP32 → INT8)
- **Accuracy Trade-off**: Expect <2% accuracy loss

### Output Formats
- **FP32**: saved_model_fp32.pth (~672 KB with base_channels=16)
- **INT8**: saved_model_int8.pth (~168 KB with base_channels=16)

**Rule**: Always maintain both FP32 and INT8 checkpoints. Document accuracy comparison.

---

## Validation & Testing Protocols

### Metrics to Track
- Training loss (combined)
- Task loss (student vs ground truth)
- Validation loss (student only)
- Distillation loss (implicit in combined loss)

### Visualization
- Training history plot (train/val loss over epochs)
### Anti-Patterns (DO NOT DO)

```python
# ✗ WRONG: Including teacher in optimizer
optimizer = torch.optim.Adam(
    list(student_model.parameters()) + list(teacher_model.parameters())
)

# ✓ CORRECT: Student parameters only
optimizer = torch.optim.Adam(student_model.parameters())
```

```python
# ✗ WRONG: Teacher in training mode
teacher_model.train()
teacher_pred = teacher_model(X)

# ✓ CORRECT: Teacher in eval mode with no_grad
teacher_model.eval()
with torch.no_grad():
    teacher_pred = teacher_model(X)
```

```python
# ✗ WRONG: Validating student with teacher
def evaluate_student_model(dataloader, student, teacher, device):
    with torch.no_grad():
        student_pred = student(X)
        teacher_pred = teacher(X)  # Don't need teacher!
        loss = distillation_loss_fn(student_pred, teacher_pred, y_true)

# ✓ CORRECT: Validate student independently
def evaluate_student_model(dataloader, student, device):
    with torch.no_grad():
        student_pred = student(X)
        loss = -(y_true * torch.log(student_pred + eps)).mean()
```

```python
# ✗ WRONG: Changing input size without adjusting model
x = torch.randn(2, 3, 4096)  # Changed from 3001!
output = student_model(x)  # Will fail dimension checks

# ✓ CORRECT: Stick to 3001 or update trimming logic
x = torch.randn(2, 3, 3001)
output = student_model(x)  # Works correctly
```

### Troubleshooting Decision Tree

**Problem: Training loss not decreasing**
1. Check learning rate (too high? try 1e-4)
2. Check gradients: `torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)`
3. Verify loss function returns scalar tensor
4. Check for NaN: `assert not torch.isnan(loss).any()`

**Problem: Validation loss >> Training loss (Overfitting)**
1. Add dropout to student model (0.2-0.3)
2. Reduce model capacity (base_channels=12 instead of 16)
3. Increase data augmentation strength
4. Reduce alpha (rely more on ground truth, less on teacher)

**Problem: Task loss high, Train loss low**
1. Student over-relies on teacher
2. Solution: Decrease alpha to 0.3 (more ground truth weight)
3. Verify ground truth labels are correct

**Problem: Both losses plateau early**
1. Learning rate too low → Increase to 1e-3
2. Model capacity too small → Increase base_channels to 24
3. Temperature too high → Decrease to 2.0

- Per-epoch loss values logged to console
- Final test loss on held-out test set

### Validation Dataset
- Use `dev` split from data.train_dev_test()
- Evaluate without teacher (student independently)
- Monitor for overfitting (val loss diverges from train loss)

**Rule**: Never use test set during training or hyperparameter tuning. Test set is final evaluation only.

---

## Common Pitfalls & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Dimension mismatch in loss | Input size not multiple of 8 | Verify XiaoNet forward() trimming to 3000, padding to 3001 |
| Teacher gradients flowing | Teacher not frozen | Check `param.requires_grad = False` after model load |
| Output shape mismatch | Encoder/decoder depth mismatch | Verify 3 levels encoder, 3 levels decoder, skip connections |
| Poor student performance | High alpha (over-reliance on teacher) | Reduce alpha to 0.3, increase ground truth weight |
| Slow training | Wrong data loader config | Ensure pin_memory=True, num_workers>0, prefetch_factor=4 |
| NaN loss | Temperature too low | Verify temperature >= 2.0, avoid division by zero in KL |

---

## Code Quality Rules

### Import Order (Notebook)
1. System/Standard library
2. Data science (numpy, pandas)
3. Visualization (matplotlib, seaborn)
4. Deep learning (torch, torchvision)
5. Seismic tools (seisbench, obspy)
6. Local modules (XiaoNet)

### Error Handling
- Wrap model loads with try-except
- Wrap data loads with try-except
- Print clear error messages with context
- Use `print("✓ Success message")` for checkpoints
- Use `print("✗ Error context")` for failures

### Logging Format
```
print(f"✓ [Component]: [Status] - [Details]")
print(f"[Component] information:")
print(f"  Parameter: {value}")
print(f"Loss: {loss.item():>7f}")
```

### Comments in Code
- Document non-obvious parameter choices
- Explain distillation-specific logic (temperature, alpha)
- Mark critical dimension dependencies
- Note any deviations from standard practices

---

## Future Extensions

### Planned Work
1. **INT8 Quantization**: Post-training quantization cell
2. **Performance Benchmarking**: FP32 vs INT8 accuracy comparison
3. **Edge Deployment**: Model export for mobile/embedded systems
4. **Ensemble Methods**: Multiple student models with teacher fusion
5. **Transfer Learning**: Fine-tune on domain-specific seismic data

### Safe Expansion Points
- **Base channels**: Increase to 24, 32 for larger capacity
- **Augmentation**: Add more transforms (time shift, amplitude scaling)
- **Ensemble**: Train multiple students with different random seeds
- **Datasets**: Validate on other seismic catalogs (STEAD, etc.)

**Rule**: All extensions must maintain 3001-sample I/O consistency and frozen teacher guarantee.

---

## References & Resources

- **Seisbench**: https://github.com/seisbench/seisbench
- **PhaseNet Paper**: Zhu et al., 2019
- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network"
- **PyTorch Quantization**: https://pytorch.org/docs/stable/quantization.html
- **U-Net Architecture**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"

---

**Document Version**: 1.0  
**Last Updated**: January 28, 2026  
**Maintained By**: XiaoNet Development Team  
**For Questions**: Refer to README.md or project issues
