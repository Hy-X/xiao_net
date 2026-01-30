"""
XiaoNet: Fast student model for seismic phase picking.

Redesigned for inference speed:
- Early aggressive downsampling (stride=4)
- ConvTranspose1d upsampling
- Additive skip connections (no concat)
- Same input/output interface as original XiaoNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class XiaoNet(nn.Module):
    def __init__(
        self,
        window_len=1000,
        in_channels=3,
        num_phases=3,
        base_channels=8,
    ):
        super().__init__()
        self.window_len = window_len
        self.in_channels = in_channels
        self.num_phases = num_phases
        self.base_channels = base_channels

        # -------- Encoder --------
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        # -------- Decoder --------
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels * 8, base_channels * 4,
                kernel_size=7, stride=4, padding=3, output_padding=3, bias=False
            ),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels * 4, base_channels * 2,
                kernel_size=7, stride=4, padding=3, output_padding=3, bias=False
            ),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels * 2, base_channels,
                kernel_size=7, stride=4, padding=3, output_padding=3, bias=False
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        # -------- Output --------
        self.output = nn.Conv1d(base_channels, num_phases, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, samples)

        Returns:
            (batch, num_phases, samples)
        """
        original_size = x.size(-1)

        # -------- Encoder --------
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bottleneck(e3)

        # -------- Decoder (additive skips) --------
        d3 = self.dec3(b)
        d3 = d3[..., :e3.size(-1)] + e3

        d2 = self.dec2(d3)
        d2 = d2[..., :e2.size(-1)] + e2

        d1 = self.dec1(d2)
        d1 = d1[..., :e1.size(-1)] + e1

        out = self.output(d1)
        out = self.softmax(out)

        # Pad back if needed
        if out.size(-1) < original_size:
            out = F.pad(out, (0, original_size - out.size(-1)))

        return out
