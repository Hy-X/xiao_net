"""
Unit tests for model architectures.
"""

import torch
import pytest
from models.small_phasenet import SmallPhaseNet


def test_small_phasenet_forward():
    """Test SmallPhaseNet forward pass."""
    model = SmallPhaseNet(
        window_len=1000,
        in_channels=3,
        num_phases=3,
        base_channels=16
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 1000)  # batch=2, channels=3, samples=1000
    output = model(x)
    
    assert output.shape == (2, 3, 1000), f"Expected shape (2, 3, 1000), got {output.shape}"


def test_small_phasenet_different_sizes():
    """Test SmallPhaseNet with different window lengths."""
    for window_len in [500, 1000, 2000]:
        model = SmallPhaseNet(
            window_len=window_len,
            in_channels=3,
            num_phases=3
        )
        x = torch.randn(1, 3, window_len)
        output = model(x)
        assert output.shape == (1, 3, window_len)


if __name__ == "__main__":
    test_small_phasenet_forward()
    test_small_phasenet_different_sizes()
    print("All tests passed!")
