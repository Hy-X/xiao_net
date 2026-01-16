"""
Loss functions package for xiao_net.
Contains distillation loss and other training objectives.
"""

from .distillation_loss import DistillationLoss

__all__ = ['DistillationLoss']
