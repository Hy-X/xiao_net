"""
Data augmentation pipeline for seismic waveforms.
Includes windowing, normalization, noise injection, and other augmentations.
"""

import numpy as np
import torch
from typing import Callable, List


class AugmentPipeline:
    """
    Pipeline for applying multiple augmentations to seismic data.
    
    Args:
        augmentations: List of augmentation functions to apply
    """
    
    def __init__(self, augmentations: List[Callable]):
        self.augmentations = augmentations
    
    def __call__(self, waveform):
        """
        Apply all augmentations in sequence.
        
        Args:
            waveform: Input waveform (channels, samples) or (samples,)
        
        Returns:
            Augmented waveform
        """
        result = waveform
        for aug in self.augmentations:
            result = aug(result)
        return result


class WindowAroundSample:
    """
    Extract window around a specific sample (e.g., around P-wave arrival).
    
    Args:
        window_len: Length of window to extract
        center_sample: Sample index to center window on (if None, random)
    """
    
    def __init__(self, window_len=1000, center_sample=None):
        self.window_len = window_len
        self.center_sample = center_sample
    
    def __call__(self, waveform):
        """
        Extract window from waveform.
        
        Args:
            waveform: Input waveform (channels, samples) or (samples,)
        
        Returns:
            Windowed waveform
        """
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        
        total_len = waveform.shape[-1]
        
        if self.center_sample is None:
            # Random window
            start = np.random.randint(0, max(1, total_len - self.window_len))
        else:
            # Center around specific sample
            start = max(0, min(self.center_sample - self.window_len // 2,
                              total_len - self.window_len))
        
        end = start + self.window_len
        
        if waveform.ndim == 1:
            return waveform[start:end]
        else:
            return waveform[:, start:end]


class Normalize:
    """
    Normalize waveform to zero mean and unit variance.
    
    Args:
        method: 'zscore' (zero mean, unit std) or 'minmax' (scale to [0,1])
    """
    
    def __init__(self, method='zscore'):
        self.method = method
    
    def __call__(self, waveform):
        """
        Normalize waveform.
        
        Args:
            waveform: Input waveform (channels, samples) or (samples,)
        
        Returns:
            Normalized waveform
        """
        if isinstance(waveform, torch.Tensor):
            is_torch = True
            waveform = waveform.numpy()
        else:
            is_torch = False
        
        if self.method == 'zscore':
            # Zero mean, unit variance
            mean = np.mean(waveform, axis=-1, keepdims=True)
            std = np.std(waveform, axis=-1, keepdims=True)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            normalized = (waveform - mean) / std
        elif self.method == 'minmax':
            # Scale to [0, 1]
            min_val = np.min(waveform, axis=-1, keepdims=True)
            max_val = np.max(waveform, axis=-1, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            normalized = (waveform - min_val) / range_val
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        if is_torch:
            return torch.from_numpy(normalized)
        return normalized


class AddNoise:
    """
    Add Gaussian noise to waveform.
    
    Args:
        noise_level: Standard deviation of noise relative to signal std
    """
    
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level
    
    def __call__(self, waveform):
        """Add noise to waveform."""
        if isinstance(waveform, torch.Tensor):
            std = torch.std(waveform)
            noise = torch.randn_like(waveform) * std * self.noise_level
            return waveform + noise
        else:
            std = np.std(waveform)
            noise = np.random.randn(*waveform.shape) * std * self.noise_level
            return waveform + noise
