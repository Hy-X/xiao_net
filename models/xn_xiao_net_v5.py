import torch
import torch.nn as nn
import torch.nn.functional as F


class XiaoNetEdge(nn.Module):
    """
    Edge-optimized XiaoNet for real-time seismic phase picking.

    Optimizations vs v4 (XiaoNetFast):
    - Hardswish activation (faster on edge, quantization-friendly)
    - Reduced base channels (default 4 vs 6)
    - Quantization-ready (no inplace ops, FloatFunctional for adds)
    - Softmax output (PhaseNet-compatible, mutually exclusive phases)

    Typical deployment:
    - ONNX export: torch.onnx.export(model, x, "model.onnx", opset_version=13)
    - INT8 quantization for 2-4x CPU/NPU speedup
    """

    def __init__(self, window_len=1000, in_channels=3, num_phases=3, base_channels=4):
        super().__init__()
        self.window_len = window_len
        self.in_channels = in_channels
        self.num_phases = num_phases
        self.base_channels = base_channels

        # For quantization-aware skip connections
        self.skip_add3 = nn.quantized.FloatFunctional()
        self.skip_add2 = nn.quantized.FloatFunctional()
        self.skip_add1 = nn.quantized.FloatFunctional()

        # -------- Encoder --------
        self.enc1 = self._ds_block(in_channels, base_channels, stride=1)
        self.enc2 = self._ds_block(base_channels, base_channels * 2, stride=4)
        self.enc3 = self._ds_block(base_channels * 2, base_channels * 4, stride=4)
        self.bottleneck = self._ds_block(base_channels * 4, base_channels * 6, stride=4)

        # -------- Decoder --------
        self.dec3 = self._up_block(base_channels * 6, base_channels * 4, scale=4)
        self.dec2 = self._up_block(base_channels * 4, base_channels * 2, scale=4)
        self.dec1 = self._up_block(base_channels * 2, base_channels, scale=4)

        # -------- Output --------
        self.output = nn.Conv1d(base_channels, num_phases, kernel_size=1)

    # ------------------------------------------------------------------
    # Blocks
    # ------------------------------------------------------------------

    def _ds_block(self, in_ch, out_ch, stride=1):
        """Depthwise separable conv: Depthwise → Pointwise → BN → Hardswish"""
        return nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.Hardswish()
        )

    def _up_block(self, in_ch, out_ch, scale=4):
        """Nearest upsample + Conv1d + Hardswish (no BN for speed)"""
        return nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="nearest"),
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.Hardswish()
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        original_size = x.size(-1)

        # -------- Encoder --------
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        # -------- Decoder (quantization-friendly skip adds) --------
        d3 = self.dec3(b)[..., :e3.size(-1)]
        d3 = self.skip_add3.add(d3, e3)

        d2 = self.dec2(d3)[..., :e2.size(-1)]
        d2 = self.skip_add2.add(d2, e2)

        d1 = self.dec1(d2)[..., :e1.size(-1)]
        d1 = self.skip_add1.add(d1, e1)

        out = F.softmax(self.output(d1), dim=1)

        # Pad to original length if needed
        if out.size(-1) < original_size:
            out = F.pad(out, (0, original_size - out.size(-1)))

        return out
