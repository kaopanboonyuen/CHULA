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
from ultralytics import YOLO
from chula import CHULALoss
import os

# Dataset YAML (replace with your path)
dataset_dir = "datasets/medical-pills"
yaml_path = os.path.join(dataset_dir, "medical-pills.yaml")

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# CHULA loss with automatic features
chula_loss = CHULALoss(lambda_ce=1.0, lambda_unc=0.3, lambda_heu=0.5)

# Patch YOLO internal loss
original_loss = model.model.loss
def patched_loss(preds, targets, imgs=None):
    yolo_loss = original_loss(preds, targets, imgs)
    sigma = torch.rand_like(targets.unsqueeze(1)) * 0.1
    heuristic_masks = {0: targets==0}
    chula_term = chula_loss(preds, targets, sigma=sigma, heuristic_masks=heuristic_masks)
    return yolo_loss + 0.5*chula_term

model.model.loss = patched_loss

# Train
model.train(data=yaml_path, epochs=30, imgsz=640, batch=16)
print("✅ Training complete with CHULA loss!")