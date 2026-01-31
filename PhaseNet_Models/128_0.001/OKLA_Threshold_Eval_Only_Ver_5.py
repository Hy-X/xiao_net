#!/usr/bin/env python
# coding: utf-8

import torch
import matplotlib.pyplot as plt
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
import torchmetrics
import numpy as np
from torch.utils.data import DataLoader
from seisbench.util import worker_seeding

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained PhaseNet model
model = sbm.PhaseNet()
model.load_state_dict(torch.load("final_model.pth"))
model.to(device)

# Load the data
data = sbd.OKLA(sampling_rate=100, force=True)
train, dev, test = data.train_dev_test()

# Set up data augmentation
phase_dict = {"p_arrival_sample": "P", "s_arrival_sample": "S"}
dev_generator = sbg.GenericGenerator(dev)

augmentations = [
    sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="random", strategy="variable"),
    sbg.RandomWindow(windowlen=3001, strategy="pad"),
    sbg.Normalize(demean_axis=-1, detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(sigma=30, dim=0)
]

dev_generator.add_augmentations(augmentations)

# Set up data loader
batch_size = 512
num_workers = 18
dev_loader = DataLoader(dev_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding)

# Test loop with ROC curve and AUC using PyTorch
def test_loop(dataloader, model, device):
    model.eval()
    all_preds_p = []
    all_true_p = []
    all_preds_s = []
    all_true_s = []

    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch["X"].to(device))
            all_preds_p.append(pred[:, 0])
            all_true_p.append(batch["y"][:, 0])
            all_preds_s.append(pred[:, 1])
            all_true_s.append(batch["y"][:, 1])

    all_preds_p = torch.cat(all_preds_p, dim=0).cpu()
    all_true_p = torch.cat(all_true_p, dim=0).cpu().to(torch.int)
    all_preds_s = torch.cat(all_preds_s, dim=0).cpu()
    all_true_s = torch.cat(all_true_s, dim=0).cpu().to(torch.int)

    # Compute ROC curve and ROC AUC using PyTorch for P-phase
    fpr_p, tpr_p, thresholds_p = torchmetrics.functional.roc(all_preds_p, all_true_p, task="binary")
    roc_auc_p = torchmetrics.functional.auroc(all_preds_p, all_true_p, task="binary")

    # Compute ROC curve and ROC AUC using PyTorch for S-phase
    fpr_s, tpr_s, thresholds_s = torchmetrics.functional.roc(all_preds_s, all_true_s, task="binary")
    roc_auc_s = torchmetrics.functional.auroc(all_preds_s, all_true_s, task="binary")

    # Find index of the point closest to the top-left corner
    distances_p = np.sqrt((1 - tpr_p)**2 + fpr_p**2)
    optimal_idx_p = np.argmin(distances_p)
    optimal_threshold_p = thresholds_p[optimal_idx_p]

    distances_s = np.sqrt((1 - tpr_s)**2 + fpr_s**2)
    optimal_idx_s = np.argmin(distances_s)
    optimal_threshold_s = thresholds_s[optimal_idx_s]

    print(f"Optimal Threshold P-phase: {optimal_threshold_p}")
    print(f"Optimal Threshold S-phase: {optimal_threshold_s}")

    # Save threshold values to a file
    with open("thresholds.txt", "w") as f:
        f.write(f"Optimal Threshold P-phase: {optimal_threshold_p}\n")
        f.write(f"Optimal Threshold S-phase: {optimal_threshold_s}\n")

    return (roc_auc_p.item(), fpr_p.numpy(), tpr_p.numpy()), (roc_auc_s.item(), fpr_s.numpy(), tpr_s.numpy())

# Evaluate the model
(dev_roc_auc_p, fpr_p, tpr_p), (dev_roc_auc_s, fpr_s, tpr_s) = test_loop(dev_loader, model, device)

print(f"Dev ROC AUC P: {dev_roc_auc_p:.4f} | Dev ROC AUC S: {dev_roc_auc_s:.4f}")

# Plotting the ROC curve for P-phase
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_p, tpr_p, label='ROC curve P-phase (area = %0.2f)' % dev_roc_auc_p)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - P-phase')
plt.legend(loc="lower right")
plt.savefig("roc_curve_p.png")
plt.show()

# Plotting the ROC curve for S-phase
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_s, tpr_s, label='ROC curve S-phase (area = %0.2f)' % dev_roc_auc_s)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - S-phase')
plt.legend(loc="lower right")
plt.savefig("roc_curve_s.png")
plt.show()

