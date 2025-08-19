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

from .loss import CHULALoss
__all__ = ["CHULALoss"]