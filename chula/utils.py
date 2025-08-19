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
#  📄 Paper  : (add link when available)
#
#  © 2025 Teerapong Panboonyuen — MIT License (see LICENSE)
# =============================================================================

"""Utility functions for CHULA.

Includes:
    - compute_class_weights: Automatic calculation of class weights
      from YOLO-style dataset labels (supports binary and multi-class).
"""

from __future__ import annotations

import os
from glob import glob
import torch


def compute_class_weights(dataset_dir: str, num_classes: int | None = None):
    """Compute class weights from YOLO dataset label `.txt` files.

    Args:
        dataset_dir (str): Root dataset directory (expects `labels/train/`).
        num_classes (int, optional): Number of classes. If None, will infer from labels.

    Returns:
        torch.Tensor: Normalized class weights (higher weight = rarer class).
    """
    # Gather label files (YOLO format)
    label_files = glob(os.path.join(dataset_dir, "labels/train/**/*.txt"), recursive=True)
    if len(label_files) == 0:
        raise FileNotFoundError(f"No YOLO labels found in {dataset_dir}/labels/train/")

    # Count instances
    all_classes = []
    for f in label_files:
        with open(f) as file:
            for line in file:
                class_id = int(line.strip().split()[0])
                all_classes.append(class_id)

    if not all_classes:
        raise ValueError("No class instances found in dataset labels.")

    max_class_id = max(all_classes)
    if num_classes is None:
        num_classes = max_class_id + 1

    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    for cid in all_classes:
        if cid < num_classes:
            class_counts[cid] += 1

    # Avoid div-by-zero
    class_counts[class_counts == 0] = 1.0

    # Inverse frequency weighting
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes  # normalize

    return weights


__all__ = ["compute_class_weights"]