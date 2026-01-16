"""
Unit tests for loss functions.
"""

import torch
from loss.xn_distillation_loss import DistillationLoss


def test_distillation_loss_hard_labels():
    """Test distillation loss with only hard labels."""
    criterion = DistillationLoss(alpha=0.0)  # Only hard labels
    
    batch_size, num_phases, samples = 2, 3, 1000
    student_logits = torch.randn(batch_size, num_phases, samples)
    labels = torch.randint(0, num_phases, (batch_size, samples))
    
    loss = criterion(student_logits, labels)
    assert loss.item() > 0, "Loss should be positive"


def test_distillation_loss_with_teacher():
    """Test distillation loss with teacher predictions."""
    criterion = DistillationLoss(alpha=0.5, temperature=4.0)
    
    batch_size, num_phases, samples = 2, 3, 1000
    student_logits = torch.randn(batch_size, num_phases, samples)
    teacher_logits = torch.randn(batch_size, num_phases, samples)
    labels = torch.randint(0, num_phases, (batch_size, samples))
    
    loss = criterion(student_logits, labels, teacher_logits)
    assert loss.item() > 0, "Loss should be positive"


if __name__ == "__main__":
    test_distillation_loss_hard_labels()
    test_distillation_loss_with_teacher()
    print("All loss tests passed!")
