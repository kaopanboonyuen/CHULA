"""
Evaluation Metrics
===================

Supports mIoU and pixel-wise accuracy for segmentation evaluation.

Author: Teerapong Panboonyuen
"""

import torch
import numpy as np

def compute_metrics(preds, gts, num_classes=5):
    ious = []
    for pred, gt in zip(preds, gts):
        pred = pred.view(-1)
        gt = gt.view(-1)
        for cls in range(num_classes):
            pred_inds = pred == cls
            target_inds = gt == cls
            intersection = (pred_inds[target_inds]).long().sum().item()
            union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
            if union > 0:
                ious.append(intersection / union)
    return 100 * np.mean(ious)