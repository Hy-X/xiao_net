"""
Early stopping utility to prevent overfitting during training.
"""

import torch
import numpy as np
from pathlib import Path


class EarlyStopping:
    """
    Early stopping class to stop training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        checkpoint_dir: Directory to save model checkpoints
        verbose: Whether to print early stopping messages
    """
    
    def __init__(self, patience=10, min_delta=0.0, checkpoint_dir='checkpoints/', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = self.checkpoint_dir / 'best_model.pth'
    
    def __call__(self, val_loss, model, epoch):
        """
        Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            model: Model to save if improvement is found
            epoch: Current epoch number
        """
        score = -val_loss  # Negative because lower is better
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, epoch)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, epoch)
            self.counter = 0
    
    def save_checkpoint(self, model, epoch):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_score': self.best_score
        }, self.checkpoint_path)
        if self.verbose:
            print(f'Validation loss improved. Saving model to {self.checkpoint_path}')
