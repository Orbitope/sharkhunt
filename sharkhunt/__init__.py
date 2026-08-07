"""Wager-aware sequential detection of rating manipulators."""

from sharkhunt.detectors import (
    HierarchicalJointLLR,
    JointLLR,
    Observation,
    OutcomeSPRT,
    WeightedLLR,
    detector_suite,
)
from sharkhunt.engine import simulate_career, simulate_field
from sharkhunt.players import Shark, Tilter, Whale
from sharkhunt.rng import Rng

__all__ = [
    "HierarchicalJointLLR",
    "JointLLR",
    "Observation",
    "OutcomeSPRT",
    "Rng",
    "Shark",
    "Tilter",
    "Whale",
    "WeightedLLR",
    "detector_suite",
    "simulate_career",
    "simulate_field",
]
