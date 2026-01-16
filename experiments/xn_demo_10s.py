"""
Demo experiment script for 10-second window phase picking.
Example of how to run experiments with different configurations.
"""

import json
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xn_xiao_net import XiaoNet
from xn_utils import set_seed, setup_device, count_parameters, get_model_size_mb


def demo_10s_experiment():
    """Run a demo experiment with 10-second windows."""
    
    # Configuration for 10-second windows (assuming 100 Hz sampling rate)
    config = {
        'window_len': 1000,  # 10 seconds at 100 Hz
        'in_channels': 3,     # Z, N, E components
        'num_phases': 3,      # P, S, noise
        'base_channels': 16
    }
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Setup device
    device = setup_device('cuda')
    print(f"Using device: {device}")
    
    # Initialize model
    model = XiaoNet(
        window_len=config['window_len'],
        in_channels=config['in_channels'],
        num_phases=config['num_phases'],
        base_channels=config['base_channels']
    ).to(device)
    
    # Print model info
    num_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    print(f"\nModel Information:")
    print(f"  Parameters: {num_params:,}")
    print(f"  Model Size: {model_size:.2f} MB")
    print(f"  Window Length: {config['window_len']} samples")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 3, config['window_len']).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\nDemo experiment completed successfully!")


if __name__ == "__main__":
    demo_10s_experiment()
