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
        self.softmax = nn.Softmax(dim=1)
        
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
        # Store original size
        original_size = x.size(2)
        
        # Trim to multiple of 8 for stable pooling/upsampling
        # 3001 → 3000, which divides evenly: 3000 → 1500 → 750 → 375
        trimmed_size = (original_size // 8) * 8
        x_trimmed = x[:, :, :trimmed_size]
        
        # Encoder path
        enc1_out = self.enc1(x_trimmed)
        enc2_out = self.enc2(self.pool(enc1_out))
        enc3_out = self.enc3(self.pool(enc2_out))
        
        # Bottleneck
        bottleneck_out = self.bottleneck(self.pool(enc3_out))
        
        # Decoder path with skip connections
        up3 = self.upsample(bottleneck_out)
        up3 = up3[:, :, :enc3_out.size(2)]  # Crop to match enc3_out size
        dec3_out = self.dec3(torch.cat([up3, enc3_out], dim=1))
        
        up2 = self.upsample(dec3_out)
        up2 = up2[:, :, :enc2_out.size(2)]  # Crop to match enc2_out size
        dec2_out = self.dec2(torch.cat([up2, enc2_out], dim=1))
        
        up1 = self.upsample(dec2_out)
        up1 = up1[:, :, :enc1_out.size(2)]  # Crop to match enc1_out size
        dec1_out = self.dec1(torch.cat([up1, enc1_out], dim=1))
        
        # Output with softmax activation
        output = self.output(dec1_out)
        output = self.softmax(output)
        
        # Pad back to original input size if needed (3000 → 3001)
        if output.size(2) < original_size:
            pad_amount = original_size - output.size(2)
            output = F.pad(output, (0, pad_amount))
        
        return output
