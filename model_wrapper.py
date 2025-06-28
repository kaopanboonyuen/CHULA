"""
CHULA Model Wrapper
===================

Wraps any segmentation or detection model (e.g., DeepLabv3+, YOLOv12)
and applies the CHULA loss during training. Makes CHULA plug-and-play.

Author: Teerapong Panboonyuen
"""

import torch
import torch.nn as nn
from chula_loss import CHULALoss

class CHULAWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes: int, class_frequencies=None, use_uncertainty=True):
        super(CHULAWrapper, self).__init__()
        self.model = base_model
        self.use_uncertainty = use_uncertainty

        self.loss_fn = CHULALoss(
            num_classes=num_classes,
            class_frequencies=class_frequencies,
            lambda_ce=1.0,
            lambda_unc=0.3 if use_uncertainty else 0.0,
            lambda_heu=0.7
        )

        if use_uncertainty:
            self.uncertainty_branch = nn.Sequential(
                nn.Conv2d(256, 1, kernel_size=1),  # Assumes 256 features from the backbone
                nn.Sigmoid()
            )

    def forward(self, x, target=None):
        if self.training:
            logits = self.model(x)

            if self.use_uncertainty:
                features = self.model.backbone(x)['out']
                sigma = self.uncertainty_branch(features)
            else:
                sigma = None

            loss = self.loss_fn(logits, target, sigma=sigma)
            return loss
        else:
            with torch.no_grad():
                logits = self.model(x)
                prediction = torch.argmax(logits, dim=1)
                return prediction