"""
CHULA: Custom Heuristic Uncertainty-guided Loss for Accurate Land Title Deed Segmentation
==========================================================================================

Author: Teerapong Panboonyuen
Institution: Chulalongkorn University
Support: Second Century Fund (C2F) Postdoctoral Fellowship

Description:
------------
This module implements the CHULA loss — a novel combination of:

    ✓ Class-balanced Cross Entropy
    ✓ Uncertainty-aware Aleatoric Regularization
    ✓ Heuristic-based Geometric Prior Penalty

Designed for robust document segmentation and detection tasks, particularly
in challenging scanned environments such as Thai land title deeds.

The CHULA loss can be plugged into any PyTorch-based model,
including YOLOv8/v12, DeepLabV3+, and SAM.

Usage:
------
>>> loss_fn = CHULALoss(num_classes=5, class_frequencies=[0.2, 0.1, 0.25, 0.15, 0.3])
>>> loss = loss_fn(logits, targets, sigma=uncertainty_branch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CHULALoss(nn.Module):
    def __init__(self, num_classes, class_frequencies=None, lambda_ce=1.0, lambda_unc=0.3, lambda_heu=0.7):
        super(CHULALoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_ce = lambda_ce
        self.lambda_unc = lambda_unc
        self.lambda_heu = lambda_heu

        if class_frequencies is not None:
            freq = torch.tensor(class_frequencies, dtype=torch.float32)
            self.weights = 1.0 / torch.log(1.02 + freq)
            self.weights = self.weights / self.weights.sum()
        else:
            self.weights = torch.ones(num_classes)

        self.heuristic_classes = [0, 2]  # Example: 0 = PIN, 2 = LAND_SCALE

    def forward(self, logits, targets, sigma=None):
        """
        Args:
            logits (Tensor): shape [B, C, H, W]
            targets (Tensor): shape [B, H, W]
            sigma (Tensor): optional, aleatoric uncertainty branch [B, 1, H, W]
        """
        ce_loss = self.class_balanced_ce_loss(logits, targets)

        if sigma is not None:
            unc_loss = 0.5 * torch.exp(-sigma) * ce_loss + 0.5 * sigma
            unc_loss = unc_loss.mean()
        else:
            unc_loss = 0.0

        heu_loss = self.heuristic_loss(logits, targets)

        total = (
            self.lambda_ce * ce_loss.mean() +
            self.lambda_unc * unc_loss +
            self.lambda_heu * heu_loss
        )
        return total

    def class_balanced_ce_loss(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.weights.to(logits.device), reduction='none')

    def heuristic_loss(self, logits, targets):
        pred = torch.argmax(logits, dim=1)
        total_loss = 0.0

        for c in self.heuristic_classes:
            pred_mask = (pred == c).float()
            target_mask = (targets == c).float()

            pred_edge = self.sobel_filter(pred_mask)
            target_edge = self.sobel_filter(target_mask)

            loss = F.mse_loss(pred_edge, target_edge)
            total_loss += loss

        return total_loss / len(self.heuristic_classes)

    def sobel_filter(self, x):
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)

        x = x.unsqueeze(1)  # [B, 1, H, W]
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return grad.squeeze(1)