"""
XiaoNet V8: Dilated U-Net for low-latency edge inference on NVIDIA Orin Nano.

Built on V5B's encoder-decoder skeleton with targeted upgrades for receptive
field, latency, and memory while preserving the core topology:

Receptive-field expansion (zero extra parameters):
  - Dilated depthwise convolutions in encoder levels 2-4 (dilation=2)
    → effective kernel width ~2× wider at each strided stage
  - DilatedMultiScaleBlock at bottleneck (kernel=3, dilation={1,3,5})
    replaces V5B's multi-kernel SeismicFeatureExtractor (kernel={3,5,7})
    → 57% larger effective RF (11 vs 7 samples at bottleneck resolution)
      with fewer depthwise parameters (kernel=3 vs kernel=7)
  - Residual path in multi-scale block for gradient stability

Low-latency / TensorRT optimizations:
  - Plain tensor addition for skip connections (no FloatFunctional overhead)
  - Built-in Conv-BN fusion helper (fuse_conv_bn) for inference
  - ONNX export helper with constant folding and batch-dynamic axes
  - All ops are TensorRT-native (Conv1d, BN, Hardswish, Upsample nearest)

Preserved from V5B:
  - U-Net depth: 3 encoder stages (stride-4) + bottleneck → 64× downsampling
  - 3 decoder stages (upsample-4×) with additive skip connections
  - base_channels=7, Hardswish activation, Softmax output (P, S, noise)
  - Identical dimension flow (3001 → 751 → 188 → 47 → decoder → 3001)

Target: ≤ V5B parameter count, wider receptive field, lower batch-1 latency
Deployment: ONNX → TensorRT (FP16) on NVIDIA Orin Nano
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DilatedDepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution with dilation support.

    Same parameter count as a standard DSConv of equal kernel_size, but
    the dilated depthwise kernel spans a wider temporal window — expanding
    the effective receptive field at no extra cost.

    Effective kernel width = dilation * (kernel_size - 1) + 1

    Structure: Dilated-Depthwise Conv1d → Pointwise Conv1d
    """

    def __init__(self, in_channels, out_channels, kernel_size=5,
                 stride=1, dilation=1, bias=False):
        super().__init__()
        # Padding preserves temporal length when stride=1;
        # for stride>1 it mirrors the padding logic of V5B's GroupedDSConv.
        padding = ((kernel_size - 1) * dilation) // 2

        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation,
            groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1, bias=bias,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DilatedMultiScaleBlock(nn.Module):
    """
    Multi-scale feature extraction via dilated convolutions at bottleneck.

    Three parallel branches with increasing dilation rates share the same
    small kernel (k=3), producing effective receptive fields of 3, 7, and
    11 samples.  A residual connection stabilises gradient flow.

    Comparison with V5B's SeismicFeatureExtractor:
      V5B:  kernel={3,5,7}  → max effective RF =  7, depthwise params ∝ 3+5+7 = 15
      V8:   kernel=3, dil={1,3,5} → max effective RF = 11, depthwise params ∝ 3×3 =  9
    """

    def __init__(self, channels):
        super().__init__()
        # dilation=1 → effective RF = 3  (local detail)
        self.branch_d1 = nn.Sequential(
            DilatedDepthwiseSeparableConv1d(channels, channels, kernel_size=3, dilation=1),
            nn.BatchNorm1d(channels),
        )
        # dilation=3 → effective RF = 7  (mid-range context)
        self.branch_d3 = nn.Sequential(
            DilatedDepthwiseSeparableConv1d(channels, channels, kernel_size=3, dilation=3),
            nn.BatchNorm1d(channels),
        )
        # dilation=5 → effective RF = 11 (broad seismic context)
        self.branch_d5 = nn.Sequential(
            DilatedDepthwiseSeparableConv1d(channels, channels, kernel_size=3, dilation=5),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        # Residual + three dilated branches (fused on GPU in one pass)
        return x + self.branch_d1(x) + self.branch_d3(x) + self.branch_d5(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class XiaoNetV8(nn.Module):
    """
    Dilated U-Net for real-time seismic phase picking on NVIDIA Orin Nano.

    Architecture (identical depth / stride pattern to V5B):
        Encoder:   enc1 (s1) → enc2 (s4) → enc3 (s4) → bottleneck (s4)
        Decoder:   dec3 (↑4) → dec2 (↑4) → dec1 (↑4) → output (1×1)
        Skips:     e3→d3, e2→d2, e1→d1  (additive)

    Key changes vs V5B:
        enc1   kernel 7→5          (fine onset; adequate at full res)
        enc2   dilation 1→2        (effective kernel 5→9)
        enc3   dilation 1→2        (effective kernel 5→9)
        bottle dilation 1→2        (effective kernel 3→5)
        multi  kernel {3,5,7}→3    dilations {1,3,5}  (RF 7→11, fewer params)
        skips  FloatFunctional→+   (cleaner ONNX, less overhead)

    Args:
        window_len:    Input window length in samples (default 3001 for 30 s @ 100 Hz)
        in_channels:   Input seismic components (3 = Z, N, E)
        num_phases:    Output phase classes (3 = P, S, noise)
        base_channels: Channel width multiplier (7, same as V5B for Orin Nano)
    """

    def __init__(self, window_len=3001, in_channels=3, num_phases=3,
                 base_channels=7):
        super().__init__()
        self.window_len = window_len
        self.in_channels = in_channels
        self.num_phases = num_phases
        self.base_channels = base_channels

        bc = base_channels

        # -------- Encoder ------------------------------------------------
        # enc1: full resolution — no dilation (fine-grained phase onsets)
        self.enc1 = nn.Sequential(
            DilatedDepthwiseSeparableConv1d(in_channels, bc, kernel_size=5, dilation=1),
            nn.BatchNorm1d(bc),
            nn.Hardswish(inplace=True),
        )

        # enc2: 4× downsample, dilation=2 → effective kernel = 9  (V5B: 5)
        self.enc2 = nn.Sequential(
            DilatedDepthwiseSeparableConv1d(bc, bc * 2, kernel_size=5, stride=4, dilation=2),
            nn.BatchNorm1d(bc * 2),
            nn.Hardswish(inplace=True),
        )

        # enc3: 4× downsample, dilation=2 → effective kernel = 9  (V5B: 5)
        self.enc3 = nn.Sequential(
            DilatedDepthwiseSeparableConv1d(bc * 2, bc * 4, kernel_size=5, stride=4, dilation=2),
            nn.BatchNorm1d(bc * 4),
            nn.Hardswish(inplace=True),
        )

        # bottleneck: 4× downsample, dilation=2 → effective kernel = 5  (V5B: 3)
        self.bottleneck = nn.Sequential(
            DilatedDepthwiseSeparableConv1d(bc * 4, bc * 8, kernel_size=3, stride=4, dilation=2),
            nn.BatchNorm1d(bc * 8),
            nn.Hardswish(inplace=True),
        )

        # Multi-scale dilated fusion at bottleneck resolution
        self.multi_scale = DilatedMultiScaleBlock(bc * 8)

        # -------- Decoder ------------------------------------------------
        # dec3: ↑4× + refine
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            DilatedDepthwiseSeparableConv1d(bc * 8, bc * 4, kernel_size=5),
            nn.BatchNorm1d(bc * 4),
            nn.Hardswish(inplace=True),
        )

        # dec2: ↑4× + phase boundary refinement
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            DilatedDepthwiseSeparableConv1d(bc * 4, bc * 2, kernel_size=5),
            nn.BatchNorm1d(bc * 2),
            nn.Hardswish(inplace=True),
        )

        # dec1: ↑4× + final phase detection
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            DilatedDepthwiseSeparableConv1d(bc * 2, bc, kernel_size=5),
            nn.BatchNorm1d(bc),
            nn.Hardswish(inplace=True),
        )

        # -------- Output head --------------------------------------------
        self.output = nn.Conv1d(bc, num_phases, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Forward pass optimised for batch-1 inference on NVIDIA Orin Nano.

        Args:
            x: (batch, 3, samples) — ZNE waveform

        Returns:
            (batch, 3, samples) — P, S, noise probabilities
        """
        original_len = x.size(-1)

        # -------- Encoder --------
        e1 = self.enc1(x)           # (B, bc,   L)
        e2 = self.enc2(e1)          # (B, bc*2, L//4)
        e3 = self.enc3(e2)          # (B, bc*4, L//16)
        b  = self.bottleneck(e3)    # (B, bc*8, L//64)

        # -------- Bottleneck multi-scale fusion --------
        b = self.multi_scale(b)

        # -------- Decoder + skip connections --------
        d3 = self.dec3(b)[..., :e3.size(-1)] + e3
        d2 = self.dec2(d3)[..., :e2.size(-1)] + e2
        d1 = self.dec1(d2)[..., :e1.size(-1)] + e1

        # -------- Output --------
        out = self.softmax(self.output(d1))

        # Pad to match original input length if needed
        if out.size(-1) < original_len:
            out = F.pad(out, (0, original_len - out.size(-1)))

        return out

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def count_parameters(self):
        """Return (total, trainable) parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_model_info(self):
        """Detailed model metadata dictionary."""
        total, trainable = self.count_parameters()
        return {
            "model_name": "XiaoNetV8",
            "total_parameters": total,
            "trainable_parameters": trainable,
            "base_channels": self.base_channels,
            "window_len": self.window_len,
            "input_channels": self.in_channels,
            "output_phases": self.num_phases,
            "deployment_platform": "NVIDIA Orin Nano",
            "mixed_precision_ready": True,
            "tensorrt_compatible": True,
            "gpu_optimized": True,
            "key_improvement": (
                "Dilated encoder + multi-scale bottleneck for ~2× receptive "
                "field with fewer depthwise params; plain skip-adds for "
                "cleaner TensorRT graph"
            ),
        }

    # ------------------------------------------------------------------
    # Inference-time optimisation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fuse_conv_bn(self):
        """
        Fuse Conv1d + BatchNorm1d pairs in-place for inference speed-up.

        Merges each (pointwise Conv1d, BatchNorm1d) pair inside Sequential
        blocks into a single Conv1d, eliminating BN overhead at runtime.
        TensorRT performs this automatically during engine build, but
        fusing ahead of export produces a smaller, cleaner ONNX graph.

        Usage::

            model.eval()
            model.fuse_conv_bn()
            model.export_onnx("v8.onnx")
        """
        for module in self.modules():
            if not isinstance(module, nn.Sequential):
                continue
            children = list(module.named_children())
            for i in range(len(children) - 1):
                name_a, mod_a = children[i]
                name_b, mod_b = children[i + 1]
                if isinstance(mod_a, DilatedDepthwiseSeparableConv1d) and isinstance(mod_b, nn.BatchNorm1d):
                    mod_a.pointwise = torch.nn.utils.fusion.fuse_conv_bn_eval(
                        mod_a.pointwise, mod_b
                    )
                    setattr(module, name_b, nn.Identity())
        return self

    def export_onnx(self, path, opset_version=17):
        """
        Export model to ONNX for the TensorRT conversion pipeline.

        Args:
            path: Output ``.onnx`` file path.
            opset_version: ONNX opset (≥17 recommended for TensorRT 8.6+).

        Produces a graph with:
          - input  ``waveform``  shape (batch, 3, 3001)
          - output ``phases``    shape (batch, 3, 3001)
          - batch axis is dynamic; temporal axis is fixed at *window_len*.
        """
        self.eval()
        dummy = torch.randn(1, self.in_channels, self.window_len)
        if next(self.parameters()).is_cuda:
            dummy = dummy.cuda()
        torch.onnx.export(
            self, dummy, path,
            input_names=["waveform"],
            output_names=["phases"],
            dynamic_axes={"waveform": {0: "batch"}, "phases": {0: "batch"}},
            opset_version=opset_version,
            do_constant_folding=True,
        )

    def estimate_orin_nano_inference(self, batch_size=1, input_length=3001):
        """
        Rough estimate of inference performance on NVIDIA Orin Nano.

        Orin Nano specs:
          - 1024-core NVIDIA Ampere GPU (or 128-core on 4 GB SKU)
          - 8 GB / 4 GB LPDDR5
          - Up to 20 TOPS INT8 / 10 TFLOPS FP16

        Returns:
            dict with estimated FP32 / FP16 inference times and FLOP count.
        """
        total_params, _ = self.count_parameters()
        flops = total_params * input_length * 2  # simplified estimate
        tflops_peak = 20  # FP32 peak
        est_ms = (flops / (tflops_peak * 1e12)) * 1000
        return {
            "total_flops_approx": flops,
            "estimated_fp32_ms": round(est_ms, 4),
            "estimated_fp16_ms": round(est_ms / 2, 4),
            "peak_tflops_fp32": tflops_peak,
            "notes": "Actual latency depends on TensorRT optimisation level and memory bandwidth.",
        }
