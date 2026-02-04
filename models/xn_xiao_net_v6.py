"""
XiaoNet V6: Ultra-lightweight model for embedded seismic phase picking.

Optimized for Raspberry Pi 5 and similar edge devices:
- Depthwise separable convolutions for parameter efficiency
- Reduced base channels (6 instead of 8)
- Aggressive stride downsampling (stride=4)
- ConvTranspose1d upsampling with additive skip connections
- Minimal memory footprint while maintaining accuracy

Model Size: ~50-60K parameters (vs ~100K for V2)
Target: Real-time inference on Pi5 with minimal latency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution block: reduces parameters by ~8-9x.
    
    = Depthwise conv (channel-wise) + Pointwise conv (1x1)
    Instead of: in_channels * out_channels * kernel_size parameters
    Uses: in_channels * kernel_size + in_channels * out_channels parameters
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class XiaoNetV6(nn.Module):
    """
    Ultra-lightweight seismic phase picker for embedded devices (Pi5, Pi4, edge TPUs).
    
    Args:
        window_len: Input window length in samples (default: 1000)
        in_channels: Number of input channels (3 for ZNE)
        num_phases: Number of output phases (3: P, S, noise)
        base_channels: Base number of channels (default: 6 for Pi5)
    """
    
    def __init__(
        self,
        window_len=1000,
        in_channels=3,
        num_phases=3,
        base_channels=6,
    ):
        super().__init__()
        self.window_len = window_len
        self.in_channels = in_channels
        self.num_phases = num_phases
        self.base_channels = base_channels

        # -------- Encoder (aggressive downsampling) --------
        self.enc1 = nn.Sequential(
            DepthwiseSeparableConv1d(
                in_channels, base_channels,
                kernel_size=5, stride=1, padding=2, bias=False
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            DepthwiseSeparableConv1d(
                base_channels, base_channels * 2,
                kernel_size=5, stride=4, padding=2, bias=False
            ),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            DepthwiseSeparableConv1d(
                base_channels * 2, base_channels * 4,
                kernel_size=5, stride=4, padding=2, bias=False
            ),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Lightweight bottleneck (reduced depth)
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv1d(
                base_channels * 4, base_channels * 6,
                kernel_size=5, stride=4, padding=2, bias=False
            ),
            nn.BatchNorm1d(base_channels * 6),
            nn.ReLU(inplace=True),
        )

        # -------- Decoder (ConvTranspose1d upsampling) --------
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels * 6, base_channels * 4,
                kernel_size=5, stride=4, padding=2, output_padding=3, bias=False
            ),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels * 4, base_channels * 2,
                kernel_size=5, stride=4, padding=2, output_padding=3, bias=False
            ),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels * 2, base_channels,
                kernel_size=5, stride=4, padding=2, output_padding=3, bias=False
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        # -------- Output --------
        self.output = nn.Conv1d(base_channels, num_phases, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch, in_channels, samples)
        
        Returns:
            (batch, num_phases, samples) - softmax probabilities
        """
        original_size = x.size(-1)

        # -------- Encoder --------
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        # -------- Decoder with additive skip connections --------
        d3 = self.dec3(b)
        d3 = d3[..., :e3.size(-1)] + e3

        d2 = self.dec2(d3)
        d2 = d2[..., :e2.size(-1)] + e2

        d1 = self.dec1(d2)
        d1 = d1[..., :e1.size(-1)] + e1

        out = self.output(d1)
        out = self.softmax(out)

        # Pad back to original size if needed
        if out.size(-1) < original_size:
            out = F.pad(out, (0, original_size - out.size(-1)))

        return out

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def estimate_inference_time(self, input_length=3001, batch_size=1):
        """
        Rough estimate of inference time (FLOPs-based).
        Actual time depends on device and implementation.
        """
        # Estimate FLOPs (simplified)
        flops = 0
        current_len = input_length
        
        # Encoder
        flops += self.base_channels * 5 * current_len  # enc1
        current_len //= 4
        flops += self.base_channels * 2 * 5 * current_len  # enc2
        current_len //= 4
        flops += self.base_channels * 4 * 5 * current_len  # enc3
        current_len //= 4
        flops += self.base_channels * 6 * 5 * current_len  # bottleneck
        
        return flops
