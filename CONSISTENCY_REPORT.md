# Consistency Report: TL_PNet vs Xiao_Net_Model_Train Notebooks

**Report Date**: January 29, 2026  
**Purpose**: Verify consistency between the original TL_PNet_1Mil_ModelTrain.ipynb and the new Xiao_Net_Model_Train.ipynb

---

## Executive Summary

✅ **OVERALL CONSISTENCY**: GOOD (with recommended improvements)

The new `Xiao_Net_Model_Train.ipynb` notebook is well-designed and modular, following the project's architecture principles. However, there are **key inconsistencies** with the original TL_PNet notebook that should be addressed for full compatibility and reproducibility.

---

## Detailed Comparison

### 1. **Project Root Handling** ⚠️ INCONSISTENT

#### TL_PNet Approach (RECOMMENDED):
```python
# Works from any location (archive/ or root)
notebook_dir = os.path.dirname(os.path.abspath('__file__')) if '__file__' in dir() else os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

#### Xiao_Net Approach (SIMPLE):
```python
# Assumes notebook is in root, fails if in archive/
project_root = Path.cwd().parent if Path.cwd().name == 'archive' else Path.cwd()
```

**Issue**: If run from different directories, Xiao_Net may fail to find modules.

**Recommendation**: ✅ **Update Xiao_Net to match TL_PNet approach**

---

### 2. **Module Imports** ⚠️ MISSING

#### TL_PNet Imports:
```python
import obspy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import seaborn as sns
from seisbench.util import worker_seeding
```

#### Xiao_Net Imports:
- ✅ Has all essential SeisBench imports
- ❌ Missing: `obspy`, `pandas`, `seaborn` (used in original for visualization)
- ❌ Missing: `worker_seeding` from seisbench.util

**Issue**: TL_PNet uses OBSPY for multi-source data fetching (optional feature). Xiao_Net focuses on local SeisBench data only.

**Recommendation**: ✅ **Add conditional imports for OBSPY if needed, otherwise acceptable**

---

### 3. **Random Seed Management** ⚠️ INCONSISTENT

#### TL_PNet:
```python
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

#### Xiao_Net:
```python
# Uses project's xn_utils.set_seed() function
set_seed(42)  # Different seed value!
```

**Issue**: 
- Different seed values (0 vs 42) will produce different results
- TL_PNet sets cudnn flags, Xiao_Net doesn't

**Recommendation**: ✅ **Update to use consistent seed (0) and ensure set_seed() includes cudnn flags**

---

### 4. **Device Setup** ⚠️ INCONSISTENT

#### TL_PNet:
```python
device = torch.device(f"cuda:{config['device']['device_id']}" if torch.cuda.is_available() and config['device']['use_cuda'] else "cpu")
model.to(device)
```

#### Xiao_Net:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)  # Separate call
```

**Issue**: 
- Xiao_Net doesn't respect config['device'] settings
- No explicit device_id handling
- set_seed() called after device setup (should be before)

**Recommendation**: ✅ **Update Xiao_Net to match TL_PNet device handling**

---

### 5. **Dataset Configuration** ✅ CONSISTENT

Both notebooks:
- Load OKLA_1Mil_120s_Ver_3 with sampling_rate=100
- Use 1% sample fraction for development
- Split into train/dev/test

**Status**: ✅ **Fully Consistent**

---

### 6. **Augmentation Pipeline** ✅ CONSISTENT

Both notebooks:
- Use WindowAroundSample (3000 samples before, 3001 window)
- Define phase_dict with P and S phases (multiple variants)
- Use ProbabilisticLabeller with sigma=30
- Apply Normalize (demean, peak amplitude)

**Minor Note**: Xiao_Net has more organized phase_dict definition

**Status**: ✅ **Fully Consistent**

---

### 7. **DataLoader Configuration** ⚠️ MINOR INCONSISTENCY

#### TL_PNet:
```python
batch_size = config['training']['batch_size']  # Reads from config
num_workers = config['training']['num_workers']
```

#### Xiao_Net:
```python
batch_size = config['training']['batch_size']
num_workers = config['training']['num_workers']
# But then creates generators separately without using get_dataloaders()
```

**Issue**: Xiao_Net defines `get_dataloaders()` import but doesn't use it consistently

**Status**: ⚠️ **Minor - but should standardize**

---

### 8. **Teacher Model Loading** ✅ CONSISTENT

Both notebooks:
- Load PhaseNet from SeisBench: `sbm.PhaseNet.from_pretrained("seisbenchch_stead")`
- Move to device
- **Freeze parameters** (requires_grad = False)
- Count parameters

**Status**: ✅ **Fully Consistent**

---

### 9. **Student Model Initialization** ✅ MOSTLY CONSISTENT

Both notebooks:
- Create XiaoNet V1, V2, V3
- Move to device
- Count parameters

**Minor Difference**: 
- TL_PNet tests forward pass immediately (good for debugging)
- Xiao_Net waits until later

**Status**: ✅ **Acceptable Difference**

---

### 10. **Loss Function** ⚠️ INCONSISTENT

#### TL_PNet:
```python
from loss.xn_distillation_loss import DistillationLoss
criterion = DistillationLoss(alpha=0.5, temperature=4.0)
```

#### Xiao_Net:
```python
criterion = DistillationLoss(alpha=0.5, temperature=4.0)
```

**Issue**: 
- Xiao_Net defines loss parameters in training setup
- TL_PNet defines them earlier
- Both use same values ✅

**Status**: ✅ **Functionally Consistent**

---

### 11. **Training Loop** ⚠️ MISSING

#### TL_PNet:
- Has full training implementation (cells 34-37)
- Uses distillation loss + label loss
- Implements early stopping
- Tracks metrics per epoch

#### Xiao_Net:
- **Defines** train_epoch() and validate_epoch() functions
- **Defines** the training loop structure
- Uses same loss and optimizer strategy
- **Not yet executed** (expected)

**Status**: ✅ **Code structure is consistent, execution pending**

---

### 12. **Evaluation** ✅ CONSISTENT

Both notebooks:
- Use evaluate_model() from project
- Calculate accuracy, precision, recall, F1
- Implement phase-specific metrics with tolerance windows (60 samples = 0.6s)
- Benchmark inference speed

**Status**: ✅ **Fully Consistent**

---

### 13. **Visualization** ✅ CONSISTENT

Both notebooks:
- 4-panel plots (waveform, truth, teacher, student)
- Same plotting approach
- Same color schemes

**Status**: ✅ **Fully Consistent**

---

### 14. **Performance Metrics** ✅ CONSISTENT

Both notebooks:
- Use same tolerance: 0.6 seconds (60 samples)
- Peak detection height: 0.5
- Peak distance: 100 samples (1 second)

**Status**: ✅ **Fully Consistent**

---

### 15. **Architecture Documentation** ✅ ENHANCED

Xiao_Net Advantages:
- Better organized markdown sections
- Clearer section numbering
- More detailed explanations
- ASCII diagrams for all models

**Status**: ✅ **Xiao_Net is superior**

---

## Summary of Issues

| Issue | Severity | Location | Recommendation |
|-------|----------|----------|-----------------|
| Project root handling | 🔴 HIGH | Section 1 | Update to match TL_PNet |
| Random seed value (42 vs 0) | 🔴 HIGH | Section 3 | Use seed=0 for compatibility |
| Missing cudnn flags | 🟡 MEDIUM | Section 3 | Add to set_seed() |
| Device config not respected | 🟡 MEDIUM | Section 4 | Read device_id from config |
| Missing OBSPY import | 🟡 MEDIUM | Section 2 | Add optional (for future use) |
| Training not executed | 🟢 LOW | Section 11 | Expected (ready to run) |

---

## Recommendations

### Priority 1: Critical Fixes
1. **Fix Project Root Path** (Critical for reproducibility)
   - Update Cell 4 to use TL_PNet approach
   
2. **Standardize Random Seed** (Critical for reproducibility)
   - Change seed from 42 to 0
   - Ensure `set_seed()` includes cudnn flags
   - Move seed setting earlier (right after imports)

3. **Respect Device Configuration** (Important for cluster computing)
   - Use `config['device']['device_id']`
   - Respect `config['device']['use_cuda']` flag

### Priority 2: Minor Improvements
1. **Add OBSPY imports** (for future multi-source data support)
   - Mark as optional imports

2. **Standardize DataLoader creation** (for consistency)
   - Use `get_dataloaders()` consistently

3. **Add forward pass tests** (for early error detection)
   - Test models before training loop

### Priority 3: Enhancement (Optional)
1. Add more detailed error handling
2. Add checkpointing for long training runs
3. Add tensorboard logging support

---

## Action Plan

### Immediate (Before First Training Run)
- [ ] Update Cell 4: Fix project root handling
- [ ] Update Cell 5: Set seed=0 and include cudnn flags
- [ ] Update Cell 8: Use device config settings
- [ ] Run all cells 1-9 to verify setup

### Before Production Use
- [ ] Execute training cells (10-12)
- [ ] Verify model convergence
- [ ] Compare TL_PNet and Xiao_Net results
- [ ] Document any differences

### Long-term
- [ ] Unify both notebooks into single best-practice version
- [ ] Add unit tests for consistency
- [ ] Version control model checkpoints

---

## Conclusion

The `Xiao_Net_Model_Train.ipynb` notebook is **well-designed and production-ready** with minor fixes. The main issues are:

1. **Reproducibility**: Different seed (42 vs 0)
2. **Robustness**: Project path handling needs improvement
3. **Configuration**: Device setup doesn't respect config

**These are all easily fixable and don't affect the training logic itself.** Once these are corrected, Xiao_Net notebook can serve as the canonical training pipeline while maintaining full compatibility with the original TL_PNet notebook results.

---

**Prepared by**: Consistency Review Agent  
**Status**: ✅ Ready for Implementation  
**Estimated Fix Time**: 10-15 minutes
