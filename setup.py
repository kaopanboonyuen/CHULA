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

from setuptools import setup, find_packages

setup(
    name="chula",
    version="0.2.0",
    description="CHULA: Custom Heuristic Uncertainty-guided Loss for PyTorch/YOLO",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0"
    ],
    python_requires=">=3.8",
    url="https://github.com/kaopanboonyuen/chula",
    author="Teerapong Panboonyuen",
    author_email="kaopanboonyuen@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)