"""Display XiaoNetV5B (Sigmoid) model architecture."""

import sys
import torch
from models.xn_xiao_net_v5b_sigmoid import XiaoNetV5B

# Instantiate model with default parameters (base_channels=7)
model = XiaoNetV5B(window_len=3001, in_channels=3, num_phases=3, base_channels=7)

print('=' * 80)
print('XiaoNetV5B (Sigmoid) Architecture')
print('=' * 80)
print()
print('Model Configuration:')
print(f'  • Input: (batch, 3 channels, 3001 samples)')
print(f'  • Output: (batch, 3 phases, 3001 samples) [P, S, Noise]')
print(f'  • Base channels: {model.base_channels}')
print(f'  • Activation: Sigmoid (independent probabilities)')
print()

# Get model info
info = model.get_model_info()
total_params, trainable_params = model.count_parameters()

print('Model Statistics:')
print(f'  • Total parameters: {total_params:,}')
print(f'  • Trainable parameters: {trainable_params:,}')
print(f'  • GPU optimized: {info["gpu_optimized"]}')
print(f'  • TensorRT compatible: {info["tensorrt_compatible"]}')
print()

print('=' * 80)
print('Layer-by-Layer Breakdown')
print('=' * 80)
print()

# Print model structure with parameter counts
def count_layer_params(module):
    return sum(p.numel() for p in module.parameters())

print('ENCODER (Downsampling Path):')
print(f'  enc1 (3→7)       : {count_layer_params(model.enc1):>6,} params | kernel=7, stride=1 | Output: (7, 3001)')
print(f'  enc2 (7→14)      : {count_layer_params(model.enc2):>6,} params | kernel=5, stride=4 | Output: (14, 750)')
print(f'  enc3 (14→28)     : {count_layer_params(model.enc3):>6,} params | kernel=5, stride=4 | Output: (28, 187)')
print(f'  bottleneck(28→56): {count_layer_params(model.bottleneck):>6,} params | kernel=3, stride=4 | Output: (56, 46)')
print()

print('MULTI-SCALE FEATURE EXTRACTOR (at bottleneck):')
print(f'  multi_scale      : {count_layer_params(model.multi_scale):>6,} params | kernels=[3,5,7], parallel')
print()

print('DECODER (Upsampling Path + Skip Connections):')
print(f'  dec3 (56→28)     : {count_layer_params(model.dec3):>6,} params | upsample×4, kernel=5 | + skip from enc3')
print(f'  dec2 (28→14)     : {count_layer_params(model.dec2):>6,} params | upsample×4, kernel=5 | + skip from enc2')
print(f'  dec1 (14→7)      : {count_layer_params(model.dec1):>6,} params | upsample×4, kernel=5 | + skip from enc1')
print()

print('OUTPUT:')
print(f'  output (7→3)     : {count_layer_params(model.output):>6,} params | 1×1 conv')
print(f'  sigmoid          : activation (per-channel independence)')
print()

print('=' * 80)
print('Data Flow')
print('=' * 80)
print()
print('Input (3, 3001)')
print('     ↓')
print('enc1 → (7, 3001)  ──────────────────────────┐')
print('     ↓                                       │')
print('enc2 → (14, 750)  ─────────────────┐        │')
print('     ↓                              │        │')
print('enc3 → (28, 187)  ───────┐         │        │')
print('     ↓                    │         │        │')
print('bottleneck → (56, 46)    │         │        │')
print('     ↓                    │         │        │')
print('multi_scale → (56, 46)   │         │        │')
print('     ↓                    │         │        │')
print('dec3 → (28, 187) ←───────┘         │        │')
print('     ↓                              │        │')
print('dec2 → (14, 750) ←─────────────────┘        │')
print('     ↓                                       │')
print('dec1 → (7, 3001) ←──────────────────────────┘')
print('     ↓')
print('output + sigmoid → (3, 3001)')
print()

print('=' * 80)
print('Key Features')
print('=' * 80)
print()
print('✓ Grouped Depthwise Separable Convolutions (parameter efficient)')
print('✓ Multi-scale feature extraction at bottleneck (kernels 3, 5, 7)')
print('✓ Skip connections for precise localization (U-Net style)')
print('✓ Sigmoid activation → independent P, S, Noise probabilities')
print('✓ GPU-optimized for NVIDIA Orin Nano deployment')
print('✓ TensorRT ready for 2-3× speedup')
print()
