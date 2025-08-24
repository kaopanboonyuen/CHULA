"""
Evaluation Metrics
===================

Supports semantic mIoU and instance segmentation metrics:
AP@0.25, AP@0.50, AP@0.75, AP@0.90, Precision, Recall, F1‑score, Accuracy.

Author: Teerapong Panboonyuen
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_metrics(preds, gts, num_classes=5):
    """Semantic segmentation: mean IoU."""
    ious = []
    for pred, gt in zip(preds, gts):
        pred = pred.view(-1)
        gt = gt.view(-1)
        for cls in range(num_classes):
            pred_inds = pred == cls
            target_inds = gt == cls
            inter = (pred_inds & target_inds).sum().item()
            union = pred_inds.sum().item() + target_inds.sum().item() - inter
            if union > 0:
                ious.append(inter / union)
    return 100 * np.mean(ious) if ious else 0.0

def instance_segmentation_metrics(pred_instances, gt_instances, iou_thresholds=[0.25, 0.5, 0.75, 0.9]):
    """
    Compute instance segmentation evaluation metrics across IoU thresholds:
    AP, AR, Precision, Recall, F1, Accuracy.
    """
    stats = {thr: {'tp': 0, 'fp': 0, 'fn': 0} for thr in iou_thresholds}
    total_gt = 0

    for pred_list, gt_list in zip(pred_instances, gt_instances):
        total_gt += len(gt_list)
        if not pred_list and not gt_list:
            continue
        for thr in iou_thresholds:
            # Initialize counts
            tp = fp = fn = 0
            if not gt_list:
                fp = len(pred_list)
            elif not pred_list:
                fn = len(gt_list)
            else:
                iou = torch.zeros((len(gt_list), len(pred_list)))
                for i, gt in enumerate(gt_list):
                    for j, pred in enumerate(pred_list):
                        inter = (gt & pred).sum().item()
                        union = (gt | pred).sum().item()
                        if union > 0:
                            iou[i, j] = inter / union
                gt_idx, pred_idx = linear_sum_assignment(-iou.numpy())
                matched_iou = iou[gt_idx, pred_idx]
                tp = (matched_iou >= thr).sum().item()
                fp = len(pred_list) - tp
                fn = len(gt_list) - tp

            stats[thr]['tp'] += tp
            stats[thr]['fp'] += fp
            stats[thr]['fn'] += fn

    results = {}
    for thr, s in stats.items():
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        ap = precision * recall  # Simplified AP approximation
        ar = recall
        accuracy = tp / (total_gt + 1e-6)
        results[thr] = {
            'AP': 100 * ap,
            'AR': 100 * ar,
            'Precision': 100 * precision,
            'Recall': 100 * recall,
            'F1': 100 * f1,
            'Accuracy': 100 * accuracy
        }

    return results