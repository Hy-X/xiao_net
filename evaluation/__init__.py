"""
Evaluation package for xiao_net.
Contains metrics and evaluation functions for seismic phase picking.
"""

from .evaluate import evaluate_model, compute_metrics

__all__ = ['evaluate_model', 'compute_metrics']
