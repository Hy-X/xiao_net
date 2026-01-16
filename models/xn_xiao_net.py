"""
Small PhaseNet student model for seismic phase picking.
Lightweight U-Net-style architecture designed for edge deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class XiaoNet(nn.Module):
    """
    Lightweight student network for seismic phase picking.
    
    Architecture: U-Net style encoder-decoder with skip connections.
    Designed to be much smaller than full PhaseNet while maintaining accuracy.
    
    Args:
        window_len: Length of input seismic window (samples)
        in_channels: Number of input channels (typically 3 for ZNE)
        num_phases: Number of output phases (typically 3: P, S, noise)
        base_channels: Base number of channels (controls model size)
    """
    
    def __init__(self, window_len=1000, in_channels=3, num_phases=3, base_channels=16):
        super(XiaoNet, self).__init__()
        self.window_len = window_len
        self.in_channels = in_channels
        self.num_phases = num_phases
        self.base_channels = base_channels
        
        # Encoder (downsampling path)
        self.enc1 = self._make_conv_block(in_channels, base_channels)
        self.enc2 = self._make_conv_block(base_channels, base_channels * 2)
        self.enc3 = self._make_conv_block(base_channels * 2, base_channels * 4)
        
        # Bottleneck
        self.bottleneck = self._make_conv_block(base_channels * 4, base_channels * 8)
        
        # Decoder (upsampling path)
        self.dec3 = self._make_conv_block(base_channels * 12, base_channels * 4)  # 8 + 4 from skip
        self.dec2 = self._make_conv_block(base_channels * 6, base_channels * 2)   # 4 + 2 from skip
        self.dec1 = self._make_conv_block(base_channels * 3, base_channels)        # 2 + 1 from skip
        
        # Output layer
        self.output = nn.Conv1d(base_channels, num_phases, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    
    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block with batch norm and ReLU."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, samples)
        
        Returns:
            Output tensor of shape (batch, num_phases, samples)
        """
        # Encoder path
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(self.pool(enc1_out))
        enc3_out = self.enc3(self.pool(enc2_out))
        
        # Bottleneck
        bottleneck_out = self.bottleneck(self.pool(enc3_out))
        
        # Decoder path with skip connections
        dec3_out = self.dec3(torch.cat([self.upsample(bottleneck_out), enc3_out], dim=1))
        dec2_out = self.dec2(torch.cat([self.upsample(dec3_out), enc2_out], dim=1))
        dec1_out = self.dec1(torch.cat([self.upsample(dec2_out), enc1_out], dim=1))
        
        # Output
        output = self.output(dec1_out)
        
        return output
