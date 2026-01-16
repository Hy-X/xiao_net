"""
Evaluation functions for seismic phase picking models.
Computes metrics such as precision, recall, F1-score, and picking accuracy.
"""

import torch
import numpy as np
from typing import Dict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation data
        device: Computation device
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            waveforms = batch['X'].to(device)
            labels = batch['y']
            
            # Forward pass
            outputs = model(waveforms)
            
            # Get predictions (argmax over phase dimension)
            predictions = torch.argmax(outputs, dim=1)  # (batch, samples)
            
            # Flatten for metric computation
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
    
    # Compute metrics
    metrics = compute_metrics(np.array(all_labels), np.array(all_predictions))
    
    return metrics


def compute_metrics(y_true, y_pred, num_classes=3):
    """
    Compute classification metrics for phase picking.
    
    Args:
        y_true: True labels (flattened array)
        y_pred: Predicted labels (flattened array)
        num_classes: Number of classes (phases)
    
    Returns:
        Dictionary of metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), zero_division=0
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_support': support.tolist()
    }
    
    return metrics


def compute_picking_accuracy(predicted_picks, true_picks, tolerance=10):
    """
    Compute picking accuracy within a tolerance window.
    
    Args:
        predicted_picks: List of predicted pick times (samples)
        true_picks: List of true pick times (samples)
        tolerance: Tolerance window in samples
    
    Returns:
        Accuracy metrics for picking
    """
    # TODO: Implement picking accuracy computation
    # This would match predicted picks to true picks within tolerance
    pass
