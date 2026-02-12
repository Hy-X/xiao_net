"""
Models package for xiao_net.
Contains neural network architectures for seismic phase picking.
"""

from .xn_xiao_net import XiaoNet
from .xn_xiao_net_v8 import XiaoNetV8

__all__ = ['XiaoNet', 'XiaoNetV8']
