"""
Utility functions for xiao_net.
Includes seed setting, device setup, and helper functions.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_device(device='cuda'):
    """
    Setup and return the computation device.
    
    Args:
        device: Device string ('cuda', 'cpu', or 'mps' for Apple Silicon)
    
    Returns:
        torch.device object
    """
    if device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """
    Calculate model size in megabytes.
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
