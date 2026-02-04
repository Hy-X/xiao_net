"""
XiaoNet V5-C: Context-aware lightweight model for seismic phase picking.

Based on V5 (XiaoNetEdge) but optimized with context-aware mechanisms:
- Squeeze-and-Excitation (SE) blocks for channel attention (adaptive feature weighting)
- Dilated convolutions in bottleneck for larger receptive field without pooling
- Base channels increased slightly to 6 (balancing V5-A and V7)
- Retains Pi5-friendly features (Hardswish, Depthwise Separable, Upsample)
- Softmax output for phase probability

Target: Enhanced accuracy on complex noise/phase scenarios while keeping low parameter count.
Use case: Difficult seismic environments where context is key (e.g., distinguishing noise from weak phases).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Lightweight version for 1D seismic data.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ContextConv1d(nn.Module):
    """
    Context-aware depthwise separable convolution block.
    
    Components:
    1. Depthwise Conv (Spatial feature extraction)
    2. Pointwise Conv (Channel combination)
    3. BatchNorm + Hardswish
    4. Squeeze-and-Excitation (Channel attention)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, use_se=True, bias=False):
        super().__init__()
        
        if padding is None:
            padding = (kernel_size * dilation) // 2
            
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.Hardswish()
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(out_channels)
            
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        if self.use_se:
            x = self.se(x)
        return x


class XiaoNetV5C(nn.Module):
    """
    Context-aware optimized V5 for seismic phase picking.
    
    Improvements over V5:
    1. Base channels = 6 (Medium capacity)
    2. Squeeze-and-Excitation (SE) blocks added to Encoders and Bottleneck:
       - Helps model focus on informative channels (e.g., differentiating P vs S energy)
    3. Dilated convolutions in Bottleneck:
       - Expands receptive field to capture longer-term context without losing resolution or adding params.
    4. Feature stability with BN in decoder (like V5-A).
    
    Args:
        window_len: Input window length in samples (default: 3001)
        in_channels: Number of input channels (3)
        num_phases: Number of output phases (3)
        base_channels: Base number of channels (6)
    """
    
    def __init__(self, window_len=3001, in_channels=3, num_phases=3, base_channels=6):
        super().__init__()
        self.window_len = window_len
        self.in_channels = in_channels
        self.num_phases = num_phases
        self.base_channels = base_channels

        # Quantization-ready
        self.skip_add3 = nn.quantized.FloatFunctional()
        self.skip_add2 = nn.quantized.FloatFunctional()
        self.skip_add1 = nn.quantized.FloatFunctional()

        # -------- Encoder --------
        # enc1: kernel=5, no SE (save compute on high-res)
        self.enc1 = ContextConv1d(in_channels, base_channels, kernel_size=5, stride=1, use_se=False)
        
        # enc2: kernel=5, stride=4, with SE
        self.enc2 = ContextConv1d(base_channels, base_channels * 2, kernel_size=5, stride=4, use_se=True)
        
        # enc3: kernel=5, stride=4, with SE
        self.enc3 = ContextConv1d(base_channels * 2, base_channels * 4, kernel_size=5, stride=4, use_se=True)

        # -------- Bottleneck (Dilated + Context) --------
        # Uses dilation=2 to double receptive field
        self.bottleneck = ContextConv1d(
            base_channels * 4, base_channels * 8, 
            kernel_size=3, stride=4, dilation=2, use_se=True
        )

        # -------- Decoder --------
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            ContextConv1d(base_channels * 8, base_channels * 4, kernel_size=3, use_se=False) # Skip SE in decoder for speed
        )
        
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            ContextConv1d(base_channels * 4, base_channels * 2, kernel_size=3, use_se=False)
        )
        
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            ContextConv1d(base_channels * 2, base_channels, kernel_size=3, use_se=False)
        )

        # -------- Output --------
        self.output = nn.Conv1d(base_channels, num_phases, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        original_size = x.size(-1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        # Decoder with skip connections
        d3 = self.dec3(b)[..., :e3.size(-1)]
        d3 = self.skip_add3.add(d3, e3)

        d2 = self.dec2(d3)[..., :e2.size(-1)]
        d2 = self.skip_add2.add(d2, e2)

        d1 = self.dec1(d2)[..., :e1.size(-1)]
        d1 = self.skip_add1.add(d1, e1)

        out = F.softmax(self.output(d1), dim=1)

        if out.size(-1) < original_size:
            out = F.pad(out, (0, original_size - out.size(-1)))

        return out

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_model_info(self):
        total, trainable = self.count_parameters()
        return {
            'model_name': 'XiaoNetV5C',
            'optimizations': 'Context-Aware (SE + Dilation)',
            'total_parameters': total,
            'base_channels': self.base_channels,
            'deployment_target': 'Raspberry Pi 5 (Context Heavy)',
        }
