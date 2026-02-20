#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
Knowledge Distillation Training for XiaoNet
Teacher: PhaseNet (from STEAD)
Student: XiaoNet (v2, v3, v4, or v5)
Dataset: OKLA regional seismic data
"""

# Standard library
import os
import sys
import json
import random
import multiprocessing
from pathlib import Path

# Fix macOS multiprocessing: use 'fork' instead of default 'spawn'
# to prevent DataLoader workers from re-executing the entire script
multiprocessing.set_start_method('fork', force=True)

# Scientific computing
import numpy as np
import pandas as pd
from scipy import signal

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Seismology & SeisBench
import obspy
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding

# Progress bars
from tqdm.notebook import tqdm

# Add parent directory to path for importing local modules
sys.path.append(str(Path.cwd().parent))

# XiaoNet modules
from models.xn_xiao_net_v2 import XiaoNet as XiaoNetV2
from models.xn_xiao_net_v3 import XiaoNet as XiaoNetV3
from models.xn_xiao_net_v4 import XiaoNetFast as XiaoNetV4
from models.xn_xiao_net_v5 import XiaoNetEdge as XiaoNetV5
from loss.xn_distillation_loss import DistillationLoss

print("✓ All packages loaded successfully!")


# In[ ]:


# Early stopping class definition
class EarlyStopping:
    """
    Early stopping class to stop training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum decrease in val_loss to qualify as an improvement
        checkpoint_dir: Directory to save model checkpoints
        model_name: Name of the model (used in checkpoint filename)
        verbose: Whether to print early stopping messages
    """
    
    def __init__(self, patience=10, min_delta=0.0, checkpoint_dir='checkpoints/', model_name='XiaoNet_V5b', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.checkpoint_path = self.checkpoint_dir / f'{model_name}-best-model.pth'
    
    def __call__(self, val_loss, model, epoch, optimizer=None, scheduler=None):
        """
        Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            model: Model to save if improvement is found
            epoch: Current epoch number
            optimizer: Optimizer to save state for (optional)
            scheduler: LR scheduler to save state for (optional)
        """
        if self.best_loss is None:
            # First epoch — record baseline and save checkpoint
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler)
        elif val_loss >= self.best_loss - self.min_delta:
            # No meaningful improvement
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement found — reset counter and save
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, epoch, optimizer=None, scheduler=None):
        """Save model checkpoint with optimizer, scheduler, and early stopping states."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'early_stopping': {
                'counter': self.counter,
                'best_loss': self.best_loss,
            },
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint, self.checkpoint_path)
        if self.verbose:
            print(f'Validation loss improved ({val_loss:.6f}). Saving model to {self.checkpoint_path}')
    
    def load_state(self, checkpoint):
        """Restore early stopping state from a checkpoint for training resumption."""
        es_state = checkpoint.get('early_stopping', {})
        self.best_loss = es_state.get('best_loss', None)
        self.counter = es_state.get('counter', 0)
        if self.verbose:
            print(f'Restored EarlyStopping state: best_loss={self.best_loss}, counter={self.counter}')
    
    def __repr__(self):
        return (f"EarlyStopping(patience={self.patience}, counter={self.counter}, "
                f"best_loss={self.best_loss})")


# In[ ]:


# Define utility functions
def set_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_device(device_type='cuda'):
    """Setup compute device (CUDA if available, else CPU)."""
    if device_type == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# Set random seed for reproducibility
SEED = 0
set_seed(SEED)

# Set device
device = setup_device('cuda')
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# In[ ]:


# Load configuration from config.json
config_path = Path.cwd().parent / "config.json"

if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")

with open(config_path, "r") as f:
    config = json.load(f)

print(f"Loaded configuration from: {config_path}")
print(json.dumps(config, indent=2))


# In[ ]:


print("\nLoading PhaseNet teacher model...")
model = sbm.PhaseNet.from_pretrained("stead")
model.to(device)
model.eval()  # Set to evaluation mode for teacher

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n✓ PhaseNet teacher loaded successfully!")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model on device: {next(model.parameters()).device}")


# In[ ]:


# Load OKLA dataset
print("Loading OKLA regional seismic dataset...")
data = sbd.OKLA_1Mil_120s_Ver_3(sampling_rate=100, force=True, component_order="ENZ")

# Optional: Use subset for faster experimentation
sample_fraction = config.get('data', {}).get('sample_fraction', 0.1)
if sample_fraction < 1.0:
    print(f"Sampling {sample_fraction*100}% of data for faster training...")
    # Create a random mask for sampling
    mask = np.random.random(len(data)) < sample_fraction
    data.filter(mask, inplace=True)
    print(f"Sampled dataset size: {len(data):,}")

# Split into train/dev/test
train, dev, test = data.train_dev_test()

# In[ ]:


# Magnitude filtering (with defaults)
min_magnitude = config.get('data_filter', {}).get('min_magnitude', 1.0)
max_magnitude = config.get('data_filter', {}).get('max_magnitude', 2.0)

print(f"Applying magnitude filters: {min_magnitude} < M < {max_magnitude}")

try:
    # Filter events with magnitude above the minimum
    print(f"[Data Filter]: Start - magnitude > {min_magnitude}")
    mask = data.metadata["source_magnitude"] > min_magnitude
    data.filter(mask, inplace=True)
    print(f"[Data Filter]: Applied - magnitude > {min_magnitude}, remaining samples: {len(data):,}")
except Exception as exc:
    print("[Data Filter]: Error - Failed to apply minimum magnitude filter.")
    print(f"Details: {exc}")
    raise

try:
    # Filter events with magnitude below the maximum
    print(f"[Data Filter]: Start - magnitude < {max_magnitude}")
    mask = data.metadata["source_magnitude"] < max_magnitude
    data.filter(mask, inplace=True)
    print(f"[Data Filter]: Applied - magnitude < {max_magnitude}, remaining samples: {len(data):,}")
except Exception as exc:
    print("[Data Filter]: Error - Failed to apply maximum magnitude filter.")
    print(f"Details: {exc}")
    raise

print(f"\nMagnitude filtering complete: {len(data):,} traces in range [{min_magnitude}, {max_magnitude}]")


# In[ ]:


# Dataset summary for training
print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

# Core sizes
print(f"Total dataset size: {len(data):,}")
print(f"Train size: {len(train):,}")
print(f"Validation size: {len(dev):,}")
print(f"Test size: {len(test):,}")

# Sampling configuration
sampling_rate = config.get('data', {}).get('sampling_rate', 'unknown')
window_len = config.get('data', {}).get('window_len', 'unknown')
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Window length: {window_len} samples")

# Metadata summary (if available)
if hasattr(data, 'metadata') and data.metadata is not None:
    if 'source_magnitude' in data.metadata:
        mags = data.metadata['source_magnitude']
        print(f"Magnitude min:  {mags.min():.2f}")
        print(f"Magnitude max:  {mags.max():.2f}")
        print(f"Magnitude mean: {mags.mean():.2f}")
    print(f"Metadata columns: {list(data.metadata.columns)}")

print("=" * 60)


# In[ ]:


# Split data into train/dev/test after filtering
train, dev, test = data.train_dev_test()

# Split ratios
n_total = len(train) + len(dev) + len(test)
if n_total > 0:
    print(f"Split ratios: train={len(train)/n_total:.2%}, dev={len(dev)/n_total:.2%}, test={len(test)/n_total:.2%}")


# In[ ]:


# Dataset objects (compact summary)
print("\n" + "=" * 60)
print("DATASET OBJECTS")
print("=" * 60)
print(f"Train dataset: {train}")
print(f"Dev dataset:   {dev}")
print(f"Test dataset:  {test}")
# Split ratios
n_total = len(train) + len(dev) + len(test)
if n_total > 0:
    print(f"Split ratios: train={len(train)/n_total:.2%}, dev={len(dev)/n_total:.2%}, test={len(test)/n_total:.2%}")

print("=" * 60)


# In[ ]:


# Set up data augmentation

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


# In[ ]:


# Create the data generators for training and validation
train_generator = sbg.GenericGenerator(train)
dev_generator = sbg.GenericGenerator(dev)
test_generator = sbg.GenericGenerator(test)


# In[ ]:


# Define phase lists for labeling
p_phases = [key for key, val in phase_dict.items() if val == "P"]
s_phases = [key for key, val in phase_dict.items() if val == "S"]

train_generator = sbg.GenericGenerator(train)
dev_generator = sbg.GenericGenerator(dev)
test_generator = sbg.GenericGenerator(test)

augmentations = [
    #sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="random", strategy="variable"),
    sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=1000, windowlen=2000, selection="random", strategy="variable"),
    #sbg.RandomWindow(windowlen=3001, strategy="pad"),
    sbg.RandomWindow(windowlen=1001, strategy="pad"),
    sbg.Normalize(demean_axis=-1, detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.ChangeDtype(np.float32),
    #sbg.ProbabilisticLabeller(sigma=30, dim=0),
    sbg.ProbabilisticLabeller(sigma=10, dim=0),
]

train_generator.add_augmentations(augmentations)
dev_generator.add_augmentations(augmentations)
test_generator.add_augmentations(augmentations)


# In[ ]:


# Parameters for peak detection (with defaults)
sampling_rate = config.get('peak_detection', {}).get('sampling_rate', 100)
height = config.get('peak_detection', {}).get('height', 0.5)
distance = config.get('peak_detection', {}).get('distance', 100)

print("\n" + "=" * 60)
print("PEAK DETECTION SETTINGS")
print("=" * 60)
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Height threshold: {height}")
print(f"Minimum peak distance: {distance} samples")
print("=" * 60)


# In[ ]:


# Parameters for peak detection
batch_size = config['training']['batch_size']
num_workers = config['training']['num_workers']
print(f"[DataLoader]: batch_size={batch_size}, num_workers={num_workers}")


# In[ ]:


print("\n" + "=" * 60)
print("TRAINING CONFIGURATION SUMMARY")
print("=" * 60)

# Dataset info
print("[Dataset]")
print(f"  Total samples:      {len(data):,}")
print(f"  Train/Validation/Test:     {len(train):,} / {len(dev):,} / {len(test):,}")
print(f"  Sample fraction:    {sample_fraction*100:.1f}%")

# Device
print("\n[Device]")
print(f"  Device:             {device}")

# Training hyperparameters
print("\n[Training]")
print(f"  Batch size:         {batch_size}")
print(f"  Num workers:        {num_workers}")
print(f"  Learning rate:      {config['training']['learning_rate']}")
print(f"  Epochs:             {config['training']['epochs']}")
print(f"  Patience:           {config['training']['patience']}")

# Peak detection
print("\n[Peak Detection]")
print(f"  Sampling rate:      {sampling_rate} Hz")
print(f"  Height threshold:   {height}")
print(f"  Min peak distance:  {distance} samples")

print("=" * 60)
print("Ready to start training!")
print("=" * 60)


# In[ ]:


# Load the data for machine learning

train_loader = DataLoader(train_generator,batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_seeding,pin_memory=True,prefetch_factor=4,persistent_workers=True)
test_loader = DataLoader(test_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding,pin_memory=True,prefetch_factor=4,persistent_workers=True)
val_loader = DataLoader(dev_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding,pin_memory=True,prefetch_factor=4,persistent_workers=True)


# In[ ]:


# Define loss function
def loss_fn(y_pred, y_true, eps=1e-8):
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)
    h = h.mean()
    return -h


# In[ ]:


# Learning rate and number of epochs
learning_rate = config['training']['learning_rate']
epochs = config['training']['epochs']

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# LR scheduler: reduce LR by half when val loss plateaus for 3 epochs
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    #verbose=True,
    min_lr=1e-6,
)

print("\n" + "=" * 60)
print("OPTIMIZER SETTINGS")
print("=" * 60)
print(f"Optimizer: {optimizer.__class__.__name__}")
print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6)")
print("=" * 60)


# In[ ]:


# Early stopping and checkpoint setup
checkpoint_dir = Path.cwd().parent / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Descriptive model name for checkpoint files
# Format: {architecture}_{dataset}_{window}_{version}
model_name = f"{model.__class__.__name__}_Okla_PNet_10s_sigma10_ver_1"

best_model_path = checkpoint_dir / f"{model_name}-best-model.pth"
final_model_path = checkpoint_dir / f"{model_name}-final-model.pth"
history_path = checkpoint_dir / f"{model_name}-loss_history.json"

patience = config.get('training', {}).get('patience', 5)
min_delta = config.get('training', {}).get('min_delta', 0.0)

early_stopping = EarlyStopping(
    patience=patience,
    min_delta=min_delta,
    checkpoint_dir=checkpoint_dir,
    model_name=model_name,
    verbose=True,
)

# Loss history container
history = {
    "train_loss": [],
    "val_loss": [],
    "learning_rate": [],
}

# Helper functions for saving
def save_loss_history(history_dict, path):
    with open(path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"✓ Loss history saved to {path}")


def save_final_model(model, path, epoch=None, optimizer=None, scheduler=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)
    print(f"Final model saved to {path}")

print("\n" + "=" * 60)
print("EARLY STOPPING & CHECKPOINTS")
print("=" * 60)
print(f"Model name:     {model_name}")
print(f"Checkpoint dir: {checkpoint_dir}")
print(f"Best model:     {best_model_path}")
print(f"Final model:    {final_model_path}")
print(f"History file:   {history_path}")
print(f"Patience:       {patience}")
print(f"Min delta:      {min_delta}")
print("=" * 60)


# In[ ]:


# Training loop with early stopping
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    dataset_size = len(train_loader.dataset)
    
    for batch_id, batch in enumerate(train_loader):
        # Forward pass
        pred = model(batch["X"].to(device))
        loss = loss_fn(pred, batch["y"].to(device))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Progress tracking
        if batch_id % 5 == 0:
            current = batch_id * len(batch["X"])
            print(f"  loss: {loss.item():>7f}  [{current:>5d}/{dataset_size:>5d}]")
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    history["train_loss"].append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch["X"].to(device))
            val_loss += loss_fn(pred, batch["y"].to(device)).item()
    
    avg_val_loss = val_loss / len(val_loader)
    history["val_loss"].append(avg_val_loss)
    
    # Step the LR scheduler based on validation loss
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(avg_val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    history["learning_rate"].append(current_lr)
    
    # Print epoch summary
    print(f"\nEpoch {epoch+1}/{epochs} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    print(f"  LR:         {current_lr:.2e}")
    if new_lr < current_lr:
        print(f"  ↓ LR reduced to {new_lr:.2e}")
    
    # Check early stopping (saves optimizer + scheduler state in checkpoint)
    early_stopping(avg_val_loss, model, epoch, optimizer=optimizer, scheduler=scheduler)
    
    if early_stopping.early_stop:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break
    
    print("-" * 60)

# Reload the best model weights before saving final model
# This ensures the final model has the best weights, not the last epoch's weights
print("\nLoading best model weights for final save...")
best_checkpoint = torch.load(early_stopping.checkpoint_path, weights_only=False)
model.load_state_dict(best_checkpoint['model_state_dict'])
print(f"✓ Loaded best weights from epoch {best_checkpoint['epoch'] + 1}")

# Save final model (now with best weights) and history
save_final_model(model, final_model_path, epoch=best_checkpoint['epoch'], optimizer=optimizer, scheduler=scheduler)
save_loss_history(history, history_path)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Best model saved to: {best_model_path}")
print(f"Final model saved to: {final_model_path}")
print(f"Loss history saved to: {history_path}")
print(f"Best epoch: {best_checkpoint['epoch'] + 1}")
print("=" * 60)


# In[ ]:


# In[ ]:


# Create the output folder
output_folder = "single_examples"
os.makedirs(output_folder, exist_ok=True)

# Parameters for peak detection (with defaults)
sampling_rate = config.get('peak_detection', {}).get('sampling_rate', 100)
height = config.get('peak_detection', {}).get('height', 0.5)
distance = config.get('peak_detection', {}).get('distance', 100)

print(f"✓ [Peak Detection]: sampling_rate={sampling_rate} Hz, height={height}, distance={distance}")


# In[ ]:


from scipy.signal import find_peaks


# In[ ]:


# Run the prediction 20 times
for i in range(1, 21):
    # Visualizing Predictions
    sample = test_generator[np.random.randint(len(test_generator))]

    waveform = sample["X"]  # Shape: (3, N)
    labels = sample["y"]  # Shape: (3, N)

    time_axis = np.arange(waveform.shape[1])  # Create a time axis

    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(5, 1, sharex=True, gridspec_kw={"hspace": 0.3})

    # Color setup
    channel_names = ["Channel E", "Channel N", "Channel Z"]
    waveform_colors = ['#a3b18a', '#588157', '#344e41']  # Custom colors for channels
    label_colors = ['#15616d', '#ff7d00']

    # Plot waveforms
    for j in range(3):
        axs[j].plot(time_axis, waveform[j], color=waveform_colors[j], linewidth=1.5)
        axs[j].set_title(f"{channel_names[j]} - Seismic Waveform", fontsize=12, fontweight='bold')
        axs[j].set_ylabel("Amplitude", fontsize=10)
        axs[j].grid(True, linestyle='--', alpha=0.6)
    
    #axs[0].plot(sample["X"].T)
    #axs[1].plot(sample["y"].T)

    # Find peaks in the ground truth labels
    y_p_peaks, _ = find_peaks(sample["y"][0], height=height, distance=distance)
    y_s_peaks, _ = find_peaks(sample["y"][1], height=height, distance=distance)

    # Convert ground truth peak indices to time values
    y_p_arrival_times = y_p_peaks / sampling_rate
    y_s_arrival_times = y_s_peaks / sampling_rate

    axs[3].plot(time_axis, labels[0], color=label_colors[0], linewidth=1.5, label="P-phase")
    axs[3].plot(time_axis, labels[1], color=label_colors[1], linewidth=1.5, label="S-phase")
    axs[3].plot(y_p_peaks, sample["y"][0, y_p_peaks], 'o', label='P arrivals', color='red')
    axs[3].plot(y_s_peaks, sample["y"][1, y_s_peaks], 'o', label='S arrivals', color='blue')
    axs[3].set_title("Dataset Ground Truth", fontsize=12, fontweight='bold')
    axs[3].set_ylim(0,1.1)
    axs[3].set_ylabel("Probability", fontsize=10)
    axs[3].grid(True, linestyle='--', alpha=0.6)
    axs[3].legend(fontsize=10, loc="upper left")
    
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        pred = model(torch.tensor(sample["X"], device=device).unsqueeze(0))  # Add a fake batch dimension
        pred = pred[0].cpu().numpy()

    # Extract the probability distributions for P and S phases
    p_prob = pred[0]
    s_prob = pred[1]

    # Identify peaks in the probability distributions
    p_peaks, _ = find_peaks(p_prob, height=height, distance=distance)
    s_peaks, _ = find_peaks(s_prob, height=height, distance=distance)

    # Convert peak indices to time values
    p_arrival_times = p_peaks / sampling_rate
    s_arrival_times = s_peaks / sampling_rate

    # Calculate residuals
    residual_p_arrival_times = p_arrival_times - y_p_arrival_times[:, np.newaxis]
    residual_s_arrival_times = s_arrival_times - y_s_arrival_times[:, np.newaxis]

    # Plot the probability distributions and the detected peaks
    axs[4].plot(p_prob, color=label_colors[0], linewidth=1.5,label='P-phase')
    axs[4].plot(p_peaks, p_prob[p_peaks], 'x', label='Detected P Arrival', color='red')
    axs[4].plot(s_prob, color=label_colors[1], linewidth=1.5,label='S-phase')
    axs[4].plot(s_peaks, s_prob[s_peaks], 'x', label='Detected S Arrival', color='blue')
    axs[4].set_title('Model Prediction', fontsize=12, fontweight='bold')
    axs[4].set_ylim(0,1.1)
    axs[4].grid(True, linestyle='--', alpha=0.6)
    axs[4].set_ylabel('Probability', fontsize=10)
    axs[4].legend(fontsize=10, loc="upper left")

    # Improve x-axis visibility
    axs[4].set_xlabel("Time (samples)", fontsize=11, fontweight='bold')
    axs[4].tick_params(axis='x', labelsize=10)

    #plt.tight_layout()
    plot_filename = os.path.join(output_folder, f"Okla_Model_Pred_{i:03d}_Plot.png")
    plt.savefig(plot_filename)
    plt.show()
    plt.close(fig)

    # Save the results to a text file
    results_filename = os.path.join(output_folder, f"Okla_Model_Pred_{i:03d}_Results.txt")
    with open(results_filename, "w") as f:
        f.write(f"Ground Truth P arrival times: {y_p_arrival_times}\n")
        f.write(f"Ground Truth S arrival times: {y_s_arrival_times}\n")
        f.write(f"Model Predicted P arrival times: {p_arrival_times}\n")
        f.write(f"Model Predicted S arrival times: {s_arrival_times}\n")
        f.write(f"Residual P arrival times: {residual_p_arrival_times}\n")
        f.write(f"Residual S arrival times: {residual_s_arrival_times}\n")

    # Save the parameters to a text file
    parameters_filename = os.path.join(output_folder, f"Okla_Model_Pred_{i:03d}_Parameters.txt")
    with open(parameters_filename, "w") as f:
        f.write(f"Data Sampling Rate: {sampling_rate}\n")
        f.write(f"Detection Height Parameter: {height}\n")
        f.write(f"Detection Distance Parameter: {distance}\n")


# In[ ]:


# Run the prediction 10 times
for i in range(1, 11):
    # Visualizing Predictions
    sample = test_generator[np.random.randint(len(test_generator))]

    waveform = sample["X"]  # Shape: (3, N)
    labels = sample["y"]  # Shape: (3, N)

    time_axis = np.arange(waveform.shape[1])  # Create a time axis

    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(5, 1, sharex=True, gridspec_kw={"hspace": 0.3})

    # Color setup
    channel_names = ["Channel E", "Channel N", "Channel Z"]
    waveform_colors = ['#a3b18a', '#588157', '#344e41']  # Custom colors for channels
    label_colors = ['#15616d', '#ff7d00']

    # Plot waveforms
    for j in range(3):
        axs[j].plot(time_axis, waveform[j], color=waveform_colors[j], linewidth=1.5)
        axs[j].set_title(f"{channel_names[j]} - Seismic Waveform", fontsize=12, fontweight='bold')
        axs[j].set_ylabel("Amplitude", fontsize=10)
        axs[j].grid(True, linestyle='--', alpha=0.6)
    
    #axs[0].plot(sample["X"].T)
    #axs[1].plot(sample["y"].T)

    # Find peaks in the ground truth labels
    y_p_peaks, _ = find_peaks(sample["y"][0], height=height, distance=distance)
    y_s_peaks, _ = find_peaks(sample["y"][1], height=height, distance=distance)

    # Convert ground truth peak indices to time values
    y_p_arrival_times = y_p_peaks / sampling_rate
    y_s_arrival_times = y_s_peaks / sampling_rate

    axs[3].plot(time_axis, labels[0], color=label_colors[0], linewidth=1.5, label="P-phase")
    axs[3].plot(time_axis, labels[1], color=label_colors[1], linewidth=1.5, label="S-phase")
    axs[3].plot(y_p_peaks, sample["y"][0, y_p_peaks], 'o', label='P arrivals', color='red')
    axs[3].plot(y_s_peaks, sample["y"][1, y_s_peaks], 'o', label='S arrivals', color='blue')
    axs[3].set_title("Dataset Ground Truth", fontsize=12, fontweight='bold')
    axs[3].set_ylim(0,1.1)
    axs[3].set_ylabel("Probability", fontsize=10)
    axs[3].grid(True, linestyle='--', alpha=0.6)
    axs[3].legend(fontsize=10, loc="upper left")
    
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        pred = model(torch.tensor(sample["X"], device=device).unsqueeze(0))  # Add a fake batch dimension
        pred = pred[0].cpu().numpy()

    # Extract the probability distributions for P and S phases
    p_prob = pred[0]
    s_prob = pred[1]

    # Identify peaks in the probability distributions
    p_peaks, _ = find_peaks(p_prob, height=height, distance=distance)
    s_peaks, _ = find_peaks(s_prob, height=height, distance=distance)

    # Convert peak indices to time values
    p_arrival_times = p_peaks / sampling_rate
    s_arrival_times = s_peaks / sampling_rate

    # Calculate residuals
    residual_p_arrival_times = p_arrival_times - y_p_arrival_times[:, np.newaxis]
    residual_s_arrival_times = s_arrival_times - y_s_arrival_times[:, np.newaxis]

    # Plot the probability distributions and the detected peaks
    axs[4].plot(p_prob, color=label_colors[0], linewidth=1.5,label='P-phase')
    axs[4].plot(p_peaks, p_prob[p_peaks], 'x', label='Detected P Arrival', color='red')
    axs[4].plot(s_prob, color=label_colors[1], linewidth=1.5,label='S-phase')
    axs[4].plot(s_peaks, s_prob[s_peaks], 'x', label='Detected S Arrival', color='blue')
    axs[4].set_title('Model Prediction', fontsize=12, fontweight='bold')
    axs[4].set_ylim(0,1.1)
    axs[4].grid(True, linestyle='--', alpha=0.6)
    axs[4].set_ylabel('Probability', fontsize=10)
    axs[4].legend(fontsize=10, loc="upper left")

    # Improve x-axis visibility
    axs[4].set_xlabel("Time (samples)", fontsize=11, fontweight='bold')
    axs[4].tick_params(axis='x', labelsize=10)

    #plt.tight_layout()
    plot_filename = os.path.join(output_folder, f"Okla_Model_Pred_{i:03d}_Plot.png")
    plt.savefig(plot_filename)
    plt.show()
    plt.close(fig)

    # Save the results to a text file
    results_filename = os.path.join(output_folder, f"Okla_Model_Pred_{i:03d}_Results.txt")
    with open(results_filename, "w") as f:
        f.write(f"Ground Truth P arrival times: {y_p_arrival_times}\n")
        f.write(f"Ground Truth S arrival times: {y_s_arrival_times}\n")
        f.write(f"Model Predicted P arrival times: {p_arrival_times}\n")
        f.write(f"Model Predicted S arrival times: {s_arrival_times}\n")
        f.write(f"Residual P arrival times: {residual_p_arrival_times}\n")
        f.write(f"Residual S arrival times: {residual_s_arrival_times}\n")

    # Save the parameters to a text file
    parameters_filename = os.path.join(output_folder, f"Okla_Model_Pred_{i:03d}_Parameters.txt")
    with open(parameters_filename, "w") as f:
        f.write(f"Data Sampling Rate: {sampling_rate}\n")
        f.write(f"Detection Height Parameter: {height}\n")
        f.write(f"Detection Distance Parameter: {distance}\n")


# In[ ]:


n_samples = len(test_generator)

all_residual_p_arrival_times = []
all_residual_s_arrival_times = []

# Initialize counters for ground truth P and S labels
groundtruth_p_peaks = 0
groundtruth_s_peaks = 0

# Initialize counters for residuals smaller than 0.6 (absolute value)
count_residuals_p_under_0_6 = 0
count_residuals_s_under_0_6 = 0

# Only Commenting this out to reflect the randome samples to squential samples 

for i in range(n_samples):
#for i in range(len(test_generator)):
    
    # Hongyu Xiao: randome sample works effectively
    sample = test_generator[np.random.randint(len(test_generator))]

    #sample = test_generator[i]

    # Find peaks in the ground truth labels
    y_p_peaks, _ = find_peaks(sample["y"][0], height=height, distance=distance)
    y_s_peaks, _ = find_peaks(sample["y"][1], height=height, distance=distance)

    # Update the counters
    groundtruth_p_peaks += len(y_p_peaks)
    groundtruth_s_peaks += len(y_s_peaks)

    # Convert ground truth peak indices to time values
    sampling_rate = 100  # Samples per second (100 Hz)
    y_p_arrival_times = y_p_peaks / sampling_rate
    y_s_arrival_times = y_s_peaks / sampling_rate

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        pred = model(torch.tensor(sample["X"], device=device).unsqueeze(0))  # Add a fake batch dimension
        pred = pred[0].cpu().numpy()

    # Extract the probability distributions for P and S phases
    p_prob = pred[0]
    s_prob = pred[1]

    # Identify peaks in the probability distributions
    p_peaks, _ = find_peaks(p_prob, height=height, distance=distance)
    s_peaks, _ = find_peaks(s_prob, height=height, distance=distance)

    # Convert peak indices to time values
    p_arrival_times = p_peaks / sampling_rate
    s_arrival_times = s_peaks / sampling_rate

    # Calculate residuals for P and S peaks, keeping only the smallest one by absolute value
    for y_p_time in y_p_arrival_times:
        residual_p_arrival_times = p_arrival_times - y_p_time
        if len(residual_p_arrival_times) > 0:
            min_residual_p = residual_p_arrival_times[np.argmin(np.abs(residual_p_arrival_times))]
            all_residual_p_arrival_times.append(min_residual_p)
            if np.abs(min_residual_p) < 0.6:
                count_residuals_p_under_0_6 += 1
        
    for y_s_time in y_s_arrival_times:
        residual_s_arrival_times = s_arrival_times - y_s_time
        if len(residual_s_arrival_times) > 0:
            min_residual_s = residual_s_arrival_times[np.argmin(np.abs(residual_s_arrival_times))]
            all_residual_s_arrival_times.append(min_residual_s)
            if np.abs(min_residual_s) < 0.6:
                count_residuals_s_under_0_6 += 1

# Display the total counts of ground truth P and S peaks
print(f"Total ground truth P peaks: {groundtruth_p_peaks}")
print(f"Total ground truth S peaks: {groundtruth_s_peaks}")

# Display the counts of residuals under 0.6 seconds
print(f"Total P-phase residuals under 0.6s: {count_residuals_p_under_0_6}")
print(f"Total S-phase residuals under 0.6s: {count_residuals_s_under_0_6}")

# Plot the histogram of residual P peak arrival times
plt.figure(figsize=(10, 5))
counts_p, bins_p, patches_p = plt.hist(all_residual_p_arrival_times, bins=30, color='skyblue', edgecolor='black', range=(-1, 1))

# Add labels for the number counts on each column
for count, bin_, patch in zip(counts_p, bins_p, patches_p):
    plt.text(bin_ + (bins_p[1] - bins_p[0]) / 2, count, f'{int(count)}', ha='center', va='bottom')

# Print total pick count, mean, and standard deviation
total_picks_p = len(all_residual_p_arrival_times)
mean_p = np.mean(all_residual_p_arrival_times)
std_p = np.std(all_residual_p_arrival_times)
plt.text(0.95, 0.95, f'Total Picks: {total_picks_p}', ha='right', va='top', transform=plt.gca().transAxes)
print(f"P-phase Residuals: Mean = {mean_p:.4f}, Std = {std_p:.4f}")
print(f"Total detected P picks: {total_picks_p}")

plt.title('Histogram of Residual P Peak Arrival Times')
plt.xlabel('Residual P Arrival Time (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(f"{model_name}_residual_p_histogram.png")  # Save the figure as a PNG file
plt.show()

# Plot the histogram of residual S peak arrival times
plt.figure(figsize=(10, 5))
counts_s, bins_s, patches_s = plt.hist(all_residual_s_arrival_times, bins=30, color='salmon', edgecolor='grey', range=(-1, 1))

# Add labels for the number counts on each column
for count, bin_, patch in zip(counts_s, bins_s, patches_s):
    plt.text(bin_ + (bins_s[1] - bins_s[0]) / 2, count, f'{int(count)}', ha='center', va='bottom')

# Print total pick count, mean, and standard deviation
total_picks_s = len(all_residual_s_arrival_times)
mean_s = np.mean(all_residual_s_arrival_times)
std_s = np.std(all_residual_s_arrival_times)
plt.text(0.95, 0.95, f'Total Picks: {total_picks_s}', ha='right', va='top', transform=plt.gca().transAxes)
print(f"S-phase Residuals: Mean = {mean_s:.4f}, Std = {std_s:.4f}")
print(f"Total detected S picks: {total_picks_s}")

plt.title('Histogram of Residual S Peak Arrival Times')
plt.xlabel('Residual S Arrival Time (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(f"{model_name}_residual_s_histogram.png")  # Save the figure as a PNG file
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Seaborn style
sns.set(style="whitegrid", context="talk")

# Parameters
x_min, x_max = -1, 1
bins = np.linspace(x_min, x_max, 31)  # 30 bins between -1 and 1

# Custom colors
p_color = '#15616d'
s_color = '#ff7d00'
shading_color = '#d3d3d3'  # light gray for shaded success zone

# === P-phase residuals ===
plt.figure(figsize=(12, 6))
sns.histplot(all_residual_p_arrival_times, bins=bins, kde=False, color=p_color, edgecolor='black', stat='count')

# Shaded ±0.6s zone
plt.axvspan(-0.6, 0.6, color=shading_color, alpha=0.3, label='Residual < 0.6s')

# Reference lines
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.axvline(-0.6, color='gray', linestyle=':', linewidth=1)
plt.axvline(0.6, color='gray', linestyle=':', linewidth=1)

# Stats and annotation
mean_p = np.mean(all_residual_p_arrival_times)
std_p = np.std(all_residual_p_arrival_times)
total_picks_p = len(all_residual_p_arrival_times)
fraction_p = count_residuals_p_under_0_6 / groundtruth_p_peaks

plt.text(x_min + 0.02, plt.gca().get_ylim()[1]*0.95,
         f'Total Picks: {total_picks_p}\nMean: {mean_p:.3f}s\nStd: {std_p:.3f}s\nUnder 0.6s: {count_residuals_p_under_0_6}/{groundtruth_p_peaks} ({fraction_p:.1%})',
         ha='left', va='top', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))

plt.title('Residual P Peak Arrival Times', fontsize=16)
plt.xlabel('Residual P Arrival Time (s)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xlim(x_min, x_max)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{model_name}_residual_p_histogram_shaded.png")
plt.savefig(f"{model_name}_residual_p_histogram_shaded.ps")
plt.show()


# === S-phase residuals ===
plt.figure(figsize=(12, 6))
sns.histplot(all_residual_s_arrival_times, bins=bins, kde=False, color=s_color, edgecolor='black', stat='count')

# Shaded ±0.6s zone
plt.axvspan(-0.6, 0.6, color=shading_color, alpha=0.3, label='Residual < 0.6s')

# Reference lines
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.axvline(-0.6, color='gray', linestyle=':', linewidth=1)
plt.axvline(0.6, color='gray', linestyle=':', linewidth=1)

# Stats and annotation
mean_s = np.mean(all_residual_s_arrival_times)
std_s = np.std(all_residual_s_arrival_times)
total_picks_s = len(all_residual_s_arrival_times)
fraction_s = count_residuals_s_under_0_6 / groundtruth_s_peaks

plt.text(x_min + 0.02, plt.gca().get_ylim()[1]*0.95,
         f'Total Picks: {total_picks_s}\nMean: {mean_s:.3f}s\nStd: {std_s:.3f}s\nUnder 0.6s: {count_residuals_s_under_0_6}/{groundtruth_s_peaks} ({fraction_s:.1%})',
         ha='left', va='top', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))

plt.title('Residual S Peak Arrival Times', fontsize=16)
plt.xlabel('Residual S Arrival Time (s)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xlim(x_min, x_max)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{model_name}_residual_s_histogram_shaded.png")
plt.savefig(f"{model_name}_residual_s_histogram_shaded.ps")
plt.show()


# %%
