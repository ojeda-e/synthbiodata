"""
SynthMolData - A synthetic data generator for drug discovery machine learning
==========================================================================

This package provides tools for generating synthetic drug discovery data that mimics 
real-world scenarios using realistic molecular descriptors and target properties.
"""

from .config import (
    BaseConfig,
    ADMEConfig,
    create_config,
    generate_sample_data,
)

__version__ = "0.1.0"
__all__ = [
    "BaseConfig",
    "ADMEConfig",
    "create_config",
    "generate_sample_data",
]