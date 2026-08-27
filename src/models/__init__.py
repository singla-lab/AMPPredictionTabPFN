"""
Multi-label classification strategies for TabPFN.

Five strategies from info.tex (in increasing order of label dependency modelling):
    BR  — Binary Relevance (independence baseline)
    LP  — Label Powerset (full joint, single classifier)
    CC  — Classifier Chain (sequential factorisation)
    ECC — Ensemble Classifier Chain (averaged random permutations)
    PCC — Probabilistic Classifier Chain (exact 2^L marginalisation)
"""
from src.models.binary_relevance import BinaryRelevance
from src.models.label_powerset import LabelPowerset
from src.models.classifier_chain import ClassifierChain
from src.models.ensemble_cc import EnsembleClassifierChain
from src.models.probabilistic_cc import ProbabilisticClassifierChain

__all__ = [
    "BinaryRelevance",
    "LabelPowerset",
    "ClassifierChain",
    "EnsembleClassifierChain",
    "ProbabilisticClassifierChain",
]
