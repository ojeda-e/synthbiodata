"""
Base generator class for synthetic data generation.
"""

from abc import ABC, abstractmethod
import numpy as np
import polars as pl
from faker import Faker

from synthbiodata.config.base import BaseConfig

class BaseGenerator(ABC):
    """Base class for all data generators."""
    
    def __init__(self, config: BaseConfig):
        """Initialize the generator."""
        self.config = config
        self.rng = np.random.default_rng(config.random_state)
        self.fake = Faker()
        self.fake.seed_instance(config.random_state)
    
    @abstractmethod
    def generate_data(self) -> pl.DataFrame:
        """Generate synthetic data."""
        pass
