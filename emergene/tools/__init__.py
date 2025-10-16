"""
Tools module for emergene analysis.

This module contains the main analysis functions for identifying emergent genes,
marker genes, computing gene set scores, and identifying gene modules.
"""

from ._emergene import runEMERGENE
from ._markerGene import runMarkG
from ._score import score
from ._module import identifyGeneModule

__all__ = [
    "runEMERGENE",
    "runMarkG",
    "score",
    "identifyGeneModule",
]