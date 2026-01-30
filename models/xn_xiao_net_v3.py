import torch
import torch.nn as nn
import torch.nn.functional as F

class XiaoNet(nn.Module):
    """
    Max-speed variant of XiaoNet (keeping all original naming for compatibility).

    Optimizations:
    - Kernel size = 3
    - Depthwise separable convolutions
    - Reduced base channels (default 6)
    - Additive skip connections
    """
    def __init__(self, window_len=1000, in_channels=3, num_phases=3, base_channels=6):
        super().__init__()
        self.window_len = window_len
        self.in_channels = in_channels
        self.num_phases = num_phases
        self.base_channels = base_channels

        # -------- Encoder --------
        self.enc1 = self._depthwise_block(in_channels, base_channels, stride=1)
        self.enc2 = self._depthwise_block(base_channels, base_channels * 2, stride=4)
        self.enc3 = self._depthwise_block(base_channels * 2, base_channels * 4, stride=4)
        self.bottleneck = self._depthwise_block(base_channels * 4, base_channels * 8, stride=4)

        # -------- Decoder --------
        self.dec3 = self._transpose_block(base_channels * 8, base_channels * 4, stride=4)
        self.dec2 = self._transpose_block(base_channels * 4, base_channels * 2, stride=4)
        self.dec1 = self._transpose_block(base_channels * 2, base_channels, stride=4)

        # -------- Output --------
        self.output = nn.Conv1d(base_channels, num_phases, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def _depthwise_block(self, in_ch, out_ch, stride=1):
        """Depthwise separable conv: Depthwise → Pointwise → BN → ReLU"""
        return nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _transpose_block(self, in_ch, out_ch, stride=4):
        """ConvTranspose1d + BN + ReLU"""
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_ch, out_ch,
                kernel_size=3, stride=stride,
                padding=1, output_padding=stride-1, bias=False
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
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

        # Pad to original length if needed
        if out.size(-1) < original_size:
            out = F.pad(out, (0, original_size - out.size(-1)))

        return out
