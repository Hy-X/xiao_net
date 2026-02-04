"""
XiaoNet V7: High-capacity seismic-optimized model for Pi5.

Based on V5-A but with expanded capacity for better phase discrimination:
- Increased base channels to 8 (vs V5-A's 5) for richer feature representation
- Larger kernel sizes (7-9) for improved seismic signal capture
- Better P-wave/S-wave separation with larger receptive fields
- BatchNorm in decoder for feature stability
- Depthwise separable convolutions for parameter efficiency (Pi5-compatible)
- Phase-aware skip connections
- Hardswish activation (quantization-friendly)
- Softmax output (P, S, noise probabilities)

Target: 96%+ accuracy on OKLA with ~5-6K parameters, Pi5-deployable
Use case: High-accuracy seismic phase picking on resource-constrained devices
Philosophy: Maximize accuracy while staying within Pi5 memory constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeismicConv1d(nn.Module):
    """
    Seismic-optimized depthwise separable convolution.
    
    Features:
    - Larger kernel sizes (7, 9) for better P/S wave patterns
    - Adaptive padding for phase boundary preservation
    - Depthwise separable for parameter efficiency
    - Hardswish for quantization readiness
    
    Kernel size strategy:
    - P-waves: low frequency, longer wavelength → kernel=9
    - S-waves: higher frequency, shorter wavelength → kernel=7
    - Bottleneck: medium kernel=5 for pattern synthesis
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=None, bias=False):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MultiScalePhaseDetection(nn.Module):
    """
    Multi-scale seismic feature extraction.
    
    Captures phase characteristics at different scales:
    - Large kernel (9): P-wave low-frequency envelope
    - Medium kernel (7): S-wave mid-frequency content
    - Small kernel (5): Noise and high-frequency components
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.large_kernel = nn.Sequential(
            SeismicConv1d(in_channels, out_channels, kernel_size=9, stride=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.Hardswish(inplace=True),
        )
        
        self.medium_kernel = nn.Sequential(
            SeismicConv1d(in_channels, out_channels, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.Hardswish(inplace=True),
        )
        
        self.small_kernel = nn.Sequential(
            SeismicConv1d(in_channels, out_channels, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.Hardswish(inplace=True),
        )
    
    def forward(self, x):
        """Fuse multi-scale seismic features"""
        large = self.large_kernel(x)
        medium = self.medium_kernel(x)
        small = self.small_kernel(x)
        # Additive fusion
        return large + medium + small


class XiaoNetV7(nn.Module):
    """
    High-capacity seismic-optimized model for Pi5 deployment.
    
    Improvements over V5-A:
    1. Increased base_channels (8 vs 5) for richer phase representation
    2. Larger kernel sizes (7, 9) for better seismic signal capture
    3. Multi-scale phase detection in encoder for phase discrimination
    4. Expanded bottleneck capacity for phase pattern learning
    5. BatchNorm throughout for training stability
    6. Optimized skip connections for phase boundary preservation
    7. Still Pi5-compatible with depthwise separable efficiency
    
    Args:
        window_len: Input window length in samples (default: 3001 for 30s @ 100Hz)
        in_channels: Number of input channels (3 for ZNE)
        num_phases: Number of output phases (3: P, S, noise)
        base_channels: Base number of channels (8 for V7 high-capacity)
    """
    
    def __init__(self, window_len=3001, in_channels=3, num_phases=3, base_channels=8):
        super().__init__()
        self.window_len = window_len
        self.in_channels = in_channels
        self.num_phases = num_phases
        self.base_channels = base_channels

        # For quantization-aware skip connections
        self.skip_add3 = nn.quantized.FloatFunctional()
        self.skip_add2 = nn.quantized.FloatFunctional()
        self.skip_add1 = nn.quantized.FloatFunctional()

        # -------- Encoder: Larger kernels for seismic phase capture --------
        # enc1: Direct input analysis with large kernels (9) for P-wave patterns
        self.enc1 = nn.Sequential(
            SeismicConv1d(in_channels, base_channels, kernel_size=9, stride=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.Hardswish(inplace=True),
        )

        # enc2: 4x downsampling with kernel=7, maintains P/S separation
        self.enc2 = nn.Sequential(
            SeismicConv1d(base_channels, base_channels * 2, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.Hardswish(inplace=True),
        )

        # enc3: Further downsampling with kernel=7, extracts phase patterns
        self.enc3 = nn.Sequential(
            SeismicConv1d(base_channels * 2, base_channels * 4, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm1d(base_channels * 4),
            nn.Hardswish(inplace=True),
        )

        # bottleneck: Multi-scale phase detection for improved discrimination
        self.bottleneck = MultiScalePhaseDetection(base_channels * 4, base_channels * 8)

        # -------- Decoder: Reconstruct phase boundaries with larger kernels --------
        # dec3: 4x upsample + feature refinement (kernel=7)
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            SeismicConv1d(base_channels * 8, base_channels * 4, kernel_size=7, bias=False),
            nn.BatchNorm1d(base_channels * 4),
            nn.Hardswish(inplace=True),
        )

        # dec2: 4x upsample + phase boundary refinement (kernel=7)
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            SeismicConv1d(base_channels * 4, base_channels * 2, kernel_size=7, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.Hardswish(inplace=True),
        )

        # dec1: 4x upsample + final phase detection (kernel=7)
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            SeismicConv1d(base_channels * 2, base_channels, kernel_size=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.Hardswish(inplace=True),
        )

        # -------- Output: Phase probability estimation --------
        self.output = nn.Conv1d(base_channels, num_phases, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass for seismic phase picking.
        
        Args:
            x: (batch, 3, samples) - ZNE waveform
        
        Returns:
            (batch, 3, samples) - P, S, noise probabilities
        """
        original_size = x.size(-1)

        # -------- Encoder: Extract phase features --------
        e1 = self.enc1(x)      # Full resolution P-wave pattern extraction
        e2 = self.enc2(e1)     # 1/4 resolution, P/S separation
        e3 = self.enc3(e2)     # 1/16 resolution, phase pattern compression
        b = self.bottleneck(e3)  # 1/64 resolution, multi-scale phase synthesis

        # -------- Decoder: Reconstruct phase arrivals with skip connections --------
        d3 = self.dec3(b)[..., :e3.size(-1)]
        d3 = self.skip_add3.add(d3, e3)  # Skip: phase pattern fusion

        d2 = self.dec2(d3)[..., :e2.size(-1)]
        d2 = self.skip_add2.add(d2, e2)  # Skip: P/S separation preservation

        d1 = self.dec1(d2)[..., :e1.size(-1)]
        d1 = self.skip_add1.add(d1, e1)  # Skip: fine-grained phase detection

        # -------- Output: Phase probability --------
        out = self.output(d1)
        out = self.softmax(out)

        # Pad to original size if needed
        if out.size(-1) < original_size:
            out = F.pad(out, (0, original_size - out.size(-1)))

        return out

    def count_parameters(self):
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_model_info(self):
        """Get detailed model information."""
        total, trainable = self.count_parameters()
        return {
            'model_name': 'XiaoNetV7',
            'total_parameters': total,
            'trainable_parameters': trainable,
            'base_channels': self.base_channels,
            'window_len': self.window_len,
            'input_channels': self.in_channels,
            'output_phases': self.num_phases,
            'deployment_platform': 'Raspberry Pi 5',
            'max_kernel_size': 9,
            'quantization_ready': True,
            'pi5_compatible': True,
            'expected_accuracy': '96%+',
        }

    def estimate_pi5_inference(self, batch_size=1, input_length=3001):
        """
        Estimate inference performance on Raspberry Pi 5.
        
        Pi5 specs:
        - 8-core ARM CPU (2.4 GHz)
        - ~20 GFLOPS sustained (8-bit optimized)
        - 8GB LPDDR5 memory
        
        Estimated inference time: ~40-60ms @ INT8 quantization
        """
        total_params, _ = self.count_parameters()
        
        # Rough FLOP estimation (simplified)
        flops = total_params * input_length * 2
        
        # Pi5 ARM CPU performance estimate
        gflops_sustained = 20  # INT8 optimized
        estimated_time_int8_ms = (flops / (gflops_sustained * 1e9)) * 1000
        
        return {
            'total_flops': flops,
            'estimated_inference_time_fp32_ms': estimated_time_int8_ms * 2,
            'estimated_inference_time_int8_ms': estimated_time_int8_ms,
            'sustained_gflops': gflops_sustained,
            'notes': 'Actual performance depends on background processes and quantization implementation'
        }
