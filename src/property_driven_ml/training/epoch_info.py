"""
Training data structures for property-driven machine learning.

This module defines data structures used to track training progress
and statistics during property-driven learning.
"""

from collections import namedtuple


EpochInfoTrain = namedtuple(
    "EpochInfoTrain",
    "pred_metric constr_acc constr_sec pred_loss random_loss constr_loss pred_loss_weight constr_loss_weight input_img adv_img random_img",
)
"""Training epoch information containing metrics and sample images."""

EpochInfoTest = namedtuple(
    "EpochInfoTest",
    "pred_metric constr_acc constr_sec pred_loss random_loss constr_loss input_img adv_img random_img",
)
"""Test epoch information containing metrics and sample images."""
