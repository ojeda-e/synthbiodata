"""
Core data generation module for synthetic biological data.
"""

import logging
from abc import ABC, abstractmethod
import numpy as np
import polars as pl
from faker import Faker

from ..config import BaseConfig, ADMEConfig, MolecularConfig


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


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


class MolecularGenerator(BaseGenerator):
    """Generator for molecular descriptor data."""
    
    def __init__(self, config: MolecularConfig):
        """Initialize the molecular generator."""
        super().__init__(config)
        self.config: MolecularConfig = config
    
    def _generate_molecular_descriptors(self, n_samples: int) -> dict[str, np.ndarray]:
        """Generate molecular descriptors with realistic ranges."""
        return {
            'molecular_weight': np.clip(
                self.rng.normal(self.config.mw_mean, self.config.mw_std, n_samples),
                self.config.mw_min, self.config.mw_max
            ),
            'logp': np.clip(
                self.rng.normal(self.config.logp_mean, self.config.logp_std, n_samples),
                self.config.logp_min, self.config.logp_max
            ),
            'tpsa': np.clip(
                self.rng.normal(self.config.tpsa_mean, self.config.tpsa_std, n_samples),
                self.config.tpsa_min, self.config.tpsa_max
            ),
            'hbd': self.rng.poisson(2, n_samples),
            'hba': self.rng.poisson(5, n_samples),
            'rotatable_bonds': self.rng.poisson(6, n_samples),
            'aromatic_rings': self.rng.poisson(2, n_samples),
            'formal_charge': self.rng.choice(
                [-2, -1, 0, 1, 2], size=n_samples, 
                p=[0.05, 0.15, 0.6, 0.15, 0.05]
            ),
        }
    
    def _generate_target_features(self, n_samples: int) -> dict[str, np.ndarray]:
        """Generate target protein features."""
        return {
            'target_family': self.rng.choice(
                self.config.target_families, size=n_samples, 
                p=self.config.target_family_probs
            ),
            'target_conservation': self.rng.uniform(0.3, 0.95, n_samples),
            'binding_site_size': self.rng.normal(500, 150, n_samples),
        }
    
    def _generate_chemical_fingerprints(self, n_samples: int, n_fingerprints: int = 10) -> dict[str, np.ndarray]:
        """Generate chemical fingerprints as binary features."""
        fingerprints = {}
        for i in range(n_fingerprints):
            fingerprints[f'fingerprint_{i}'] = self.rng.binomial(1, 0.3, n_samples)
        return fingerprints
    
    def _calculate_binding_probabilities(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """Calculate realistic binding probabilities based on molecular properties."""
        binding_prob = np.zeros(len(data['molecular_weight']))
        
        # Base probability
        binding_prob += 0.01
        
        # Molecular weight
        mw_mask = np.logical_and(
            data['molecular_weight'] > 300,
            data['molecular_weight'] < 500
        )
        binding_prob += 0.02 * mw_mask
        
        # LogP
        logp_mask = np.logical_and(
            data['logp'] > 1,
            data['logp'] < 4
        )
        binding_prob += 0.03 * logp_mask
        
        # TPSA
        tpsa_mask = np.logical_and(
            data['tpsa'] > 40,
            data['tpsa'] < 120
        )
        binding_prob += 0.02 * tpsa_mask
        
        # Other descriptors
        binding_prob += 0.01 * (data['hbd'] <= 5)
        binding_prob += 0.01 * (data['hba'] <= 10)
        
        # Fingerprints
        binding_prob += 0.02 * data['fingerprint_0']
        binding_prob += 0.015 * data['fingerprint_3']
        
        # Protein targets
        binding_prob += 0.01 * (data['target_family'] == 'Kinase')
        binding_prob += 0.01 * (data['target_family'] == 'GPCR')
        
        return np.clip(binding_prob, 0, 1)
    
    def _generate_labels(self, binding_prob: np.ndarray) -> np.ndarray:
        """Generate binary labels based on binding probabilities."""
        n_positive = int(self.config.n_samples * self.config.positive_ratio)
        labels = np.zeros(len(binding_prob))
        
        # Select top compounds by binding probability
        top_indices = np.argsort(binding_prob)[-n_positive:]
        labels[top_indices] = 1
        
        # Add some noise bzzz
        noise_indices = self.rng.choice(
            len(labels), 
            size=int(len(labels) * 0.05), 
            replace=False
        )
        labels[noise_indices] = 1 - labels[noise_indices]
        
        return labels.astype(int)
    
    def generate_data(self) -> pl.DataFrame:
        """Generate synthetic molecular descriptor data."""
        logger.info(f"Generating {self.config.n_samples} molecular samples...")
        if self.config.imbalanced:
            logger.info(f"Using imbalanced dataset with positive ratio: {self.config.positive_ratio:.1%}")
        
        # Generate all feature types
        molecular_data = self._generate_molecular_descriptors(self.config.n_samples)
        target_data = self._generate_target_features(self.config.n_samples)
        fingerprint_data = self._generate_chemical_fingerprints(self.config.n_samples)
        
        # Combine all features
        all_data = {**molecular_data, **target_data, **fingerprint_data}
        
        # Calculate binding probabilities and generate labels
        binding_prob = self._calculate_binding_probabilities(all_data)
        labels = self._generate_labels(binding_prob)
        
        df = pl.DataFrame(all_data)
        df = df.with_columns(pl.Series("binds_target", labels))
        
        logger.info(f"Generated {len(df)} samples")
        logger.info(f"Features: {len(df.columns) - 1}")  # Exclude target column
        logger.info(f"Positive samples: {labels.sum()} ({labels.mean():.1%})")
        
        return df


class ADMEGenerator(BaseGenerator):
    """Generator for ADME (Absorption, Distribution, Metabolism, Excretion) data."""
    
    def __init__(self, config: ADMEConfig):
        """Initialize the ADME generator."""
        super().__init__(config)
        self.config: ADMEConfig = config
    
    def generate_data(self) -> pl.DataFrame:
        """Generate synthetic ADME data."""
        logger.info(f"Generating {self.config.n_samples} ADME samples...")
        if self.config.imbalanced:
            logger.info(f"Using imbalanced dataset with positive ratio: {self.config.positive_ratio:.1%}")
        
        # Generate base features
        data = {
            'absorption': np.clip(
                self.rng.normal(
                    self.config.absorption_mean,
                    self.config.absorption_std,
                    self.config.n_samples
                ),
                0, 100
            ),
            'plasma_protein_binding': np.clip(
                self.rng.normal(
                    self.config.plasma_protein_binding_mean,
                    self.config.plasma_protein_binding_std,
                    self.config.n_samples
                ),
                0, 100
            ),
            'clearance': np.clip(
                self.rng.normal(
                    self.config.clearance_mean,
                    self.config.clearance_std,
                    self.config.n_samples
                ),
                0, None
            ),
            'half_life': np.clip(
                self.rng.normal(
                    self.config.half_life_mean,
                    self.config.half_life_std,
                    self.config.n_samples
                ),
                0, None
            ),
        }
        
        # Calculate drug bioavailability (binary target)
        bioavailability = np.logical_and.reduce([
            data['absorption'] > 50,
            data['plasma_protein_binding'] < 95,
            data['clearance'] < 8,
            data['half_life'] > 6
        ]).astype(int)
        
        # If imbalanced, adjust the labels
        if self.config.imbalanced:
            n_positive = int(self.config.n_samples * self.config.positive_ratio)
            current_positives = bioavailability.sum()
            if current_positives > n_positive:
                # Randomly set some positives to negative
                positive_indices = np.where(bioavailability == 1)[0]
                to_flip = self.rng.choice(
                    positive_indices,
                    size=int(current_positives - n_positive),
                    replace=False
                )
                bioavailability[to_flip] = 0
            elif current_positives < n_positive:
                # Randomly set some negatives to positive
                negative_indices = np.where(bioavailability == 0)[0]
                to_flip = self.rng.choice(
                    negative_indices,
                    size=int(n_positive - current_positives),
                    replace=False
                )
                bioavailability[to_flip] = 1
        

        df = pl.DataFrame(data)
        df = df.with_columns(pl.Series("good_bioavailability", bioavailability))
        
        positive_count = bioavailability.sum()
        positive_ratio = positive_count / len(bioavailability)
        logger.info(f"Generated {len(df)} samples")
        logger.info(f"Features: {len(df.columns) - 1}")  # Exclude target column
        logger.info(f"Positive samples: {positive_count} ({positive_ratio:.1%})")
        
        return df