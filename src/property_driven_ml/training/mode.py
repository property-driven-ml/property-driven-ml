"""
Possible training modes for property-driven machine learning.
"""

from enum import Enum


class Mode(Enum):
    MultiClassClassification = 1
    MultiLabelClassification = 2
    Regression = 3
