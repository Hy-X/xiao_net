"""
Distillation loss for knowledge distillation from teacher to student model.
Combines hard labels (ground truth) with soft labels (teacher predictions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    
    Loss = alpha * KL_div(teacher_logits/T, student_logits/T) + (1-alpha) * CE(student_logits, labels)
    
    Args:
        alpha: Weight for distillation loss (0 = only hard labels, 1 = only soft labels)
        temperature: Temperature for softening logits
    """
    
    def __init__(self, alpha=0.5, temperature=4.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def compute_hard_loss(self, student_logits, labels):
        """
        Compute hard label loss (ground truth supervision).
        
        Args:
            student_logits: Student model predictions (batch*samples, num_phases)
            labels: Ground truth labels (batch*samples,)
        
        Returns:
            Cross-entropy loss value
        """
        return self.ce_loss(student_logits, labels)
    
    def forward(self, student_logits, labels, teacher_logits=None):
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model predictions (batch, num_phases, samples)
            labels: Ground truth labels (batch, samples) or (batch, num_phases, samples)
            teacher_logits: Teacher model predictions (batch, num_phases, samples), optional
        
        Returns:
            Combined loss value
        """
        # Hard label loss (cross-entropy)
        if labels.dim() == 2:
            # Convert to one-hot if needed
            labels_onehot = F.one_hot(labels, num_classes=student_logits.shape[1]).float()
            labels_onehot = labels_onehot.permute(0, 2, 1)  # (batch, num_phases, samples)
        else:
            labels_onehot = labels
        
        # Reshape for loss computation
        batch_size, num_phases, samples = student_logits.shape
        student_flat = student_logits.view(batch_size * samples, num_phases)
        labels_flat = labels_onehot.view(batch_size * samples, num_phases)
        
        hard_loss = self.compute_hard_loss(student_flat, labels_flat.argmax(dim=1))
        
        # Soft label loss (knowledge distillation)
        if teacher_logits is not None and self.alpha > 0:
            teacher_flat = teacher_logits.view(batch_size * samples, num_phases)
            
            # Soften logits with temperature
            student_soft = F.log_softmax(student_flat / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_flat / self.temperature, dim=1)
            
            soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
            
            # Combine losses
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        else:
            # Only hard labels
            total_loss = hard_loss
        
        return total_loss
