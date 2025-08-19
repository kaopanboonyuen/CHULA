#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  🌸 CHULA: Custom Heuristic Uncertainty-guided Loss for Accurate Land Title
#     Deed Segmentation
#
#  🧠 Author: Teerapong Panboonyuen
#  ✉️  Email : teerapong.pa@chula.ac.th
#  🏛️ Affil.: Chulalongkorn University (C2F Postdoctoral Fellowship)
#
#  📦 Repo   : https://github.com/kaopanboonyuen/CHULA
#  📓 Colab  : https://colab.research.google.com/github/kaopanboonyuen/notebook/CHULA_LOSS_withMedicalPillsDetection.ipynb
#  📄 Paper  : (will add link again when available)
#
#  © 2025 Teerapong Panboonyuen — MIT License (see LICENSE)
# =============================================================================

"""CHULA core module.

CHULA (Custom Heuristic Uncertainty-guided Loss) enhances document AI by
combining:
  • Class-balanced cross-entropy
  • Aleatoric uncertainty modeling
  • Domain-specific heuristic priors (edge/structure consistency)

This header is intended for all Python files in the CHULA project.

Example:
    from chula.loss import CHULALoss
    loss = CHULALoss(lambda_ce=1.0, lambda_unc=0.3, lambda_heu=0.5)

Citation:
    @article{panboonyuen2025chula,
      title={CHULA: Custom Heuristic Uncertainty-guided Loss for Accurate Land Title Deed Segmentation},
      author={Panboonyuen, Teerapong},
      year={2025}
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CHULALoss(nn.Module):
    """
    CHULA: Custom Heuristic Uncertainty-guided Loss
    - Auto class weight calculation
    - Auto-detect binary/multi-class
    - Optional uncertainty branch
    """
    def __init__(self, class_weights=None, lambda_ce=1.0, lambda_unc=0.5, lambda_heu=0.5, auto_weights=True):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_unc = lambda_unc
        self.lambda_heu = lambda_heu
        self.class_weights = class_weights
        self.auto_weights = auto_weights

    def forward(self, logits, targets, sigma=None, heuristic_masks=None):
        num_classes = logits.shape[1]

        if self.auto_weights and self.class_weights is None:
            self.class_weights = self.compute_class_weights(targets, num_classes).to(logits.device)

        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)

        unc_loss = 0.0
        if sigma is not None:
            unc_loss = 0.5 * torch.exp(-sigma) * ce_loss + 0.5 * sigma.mean()

        heu_loss = 0.0
        if heuristic_masks is not None:
            pred_probs = F.softmax(logits, dim=1)
            for class_id, mask in heuristic_masks.items():
                pred_edge = self.soft_edge(pred_probs[:, class_id])
                target_edge = self.soft_edge(mask)
                heu_loss += ((pred_edge - target_edge)**2).mean()

        return self.lambda_ce*ce_loss + self.lambda_unc*unc_loss + self.lambda_heu*heu_loss

    @staticmethod
    def soft_edge(x):
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_x.transpose(2,3)
        edge_x = F.conv2d(x.unsqueeze(1), sobel_x, padding=1)
        edge_y = F.conv2d(x.unsqueeze(1), sobel_y, padding=1)
        return torch.sqrt(edge_x**2 + edge_y**2).squeeze(1)

    @staticmethod
    def compute_class_weights(targets, num_classes):
        counts = torch.zeros(num_classes)
        for c in range(num_classes):
            counts[c] = (targets == c).sum()
        counts = torch.where(counts==0, torch.ones_like(counts), counts)
        weights = 1.0 / torch.log(1.0 + counts)
        return weights