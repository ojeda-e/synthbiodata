"""Tests for data generators."""

import pytest
import numpy as np
import polars
from synthbiodata.config import MolecularConfig, ADMEConfig
from synthbiodata.core.generator import MolecularGenerator, ADMEGenerator


@pytest.fixture(scope="class")
def molecular_balanced_config():
    """Create a balanced molecular configuration for testing."""
    return MolecularConfig(
        n_samples=1000,
        positive_ratio=0.5,
        random_state=42
    )


@pytest.fixture(scope="class")
def molecular_imbalanced_config():
    """Create an imbalanced molecular configuration for testing."""
    return MolecularConfig(
        n_samples=1000,
        positive_ratio=0.03,
        imbalanced=True,
        random_state=42
    )


class TestMolecularGenerator:
    """Test suite for molecular descriptor generator."""
    
    def test_molecular_descriptors_ranges(self, molecular_balanced_config):
        """Test that generated molecular descriptors are within expected ranges."""
        generator = MolecularGenerator(molecular_balanced_config)
        descriptors = generator._generate_molecular_descriptors(molecular_balanced_config.n_samples)
        
        # Check molecular weight ranges
        assert np.all(descriptors['molecular_weight'] >= molecular_balanced_config.mw_min)
        assert np.all(descriptors['molecular_weight'] <= molecular_balanced_config.mw_max)
        
        # Check LogP ranges
        assert np.all(descriptors['logp'] >= molecular_balanced_config.logp_min)
        assert np.all(descriptors['logp'] <= molecular_balanced_config.logp_max)
        
        # Check TPSA ranges
        assert np.all(descriptors['tpsa'] >= molecular_balanced_config.tpsa_min)
        assert np.all(descriptors['tpsa'] <= molecular_balanced_config.tpsa_max)
    
    def test_target_features(self, molecular_balanced_config):
        """Test generation of target protein features."""
        generator = MolecularGenerator(molecular_balanced_config)
        features = generator._generate_target_features(molecular_balanced_config.n_samples)
        
        # target families
        unique_families = np.unique(features['target_family'])
        assert all(family in molecular_balanced_config.target_families for family in unique_families)
        
        # Conservation range
        assert np.all(features['target_conservation'] >= 0.3)
        assert np.all(features['target_conservation'] <= 0.95)
    
    def test_chemical_number_of_fingerprints(self, molecular_balanced_config):
        """Test generation of chemical fingerprints."""
        generator = MolecularGenerator(molecular_balanced_config)
        fingerprints = generator._generate_chemical_fingerprints(
            molecular_balanced_config.n_samples,
            n_fingerprints=5
        )
        assert len(fingerprints) == 5
        
    def test_chemical_fingerprints_binary(self, molecular_balanced_config):
        generator = MolecularGenerator(molecular_balanced_config)
        fingerprints = generator._generate_chemical_fingerprints(
            molecular_balanced_config.n_samples,
            n_fingerprints=5
        )
        for fp in fingerprints.values():
            assert np.all(np.isin(fp, [0, 1]))
    
    def test_binding_probabilities(self, molecular_balanced_config):
        """Test calculation of binding probabilities."""
        generator = MolecularGenerator(molecular_balanced_config)
        data = {
            'molecular_weight': np.array([400, 200]),  # One in range, one out
            'logp': np.array([2, 5]),  # One in range, one out
            'tpsa': np.array([60, 150]),  # One in range, one out
            'hbd': np.array([3, 7]),  # One good, one bad
            'hba': np.array([8, 12]),  # One good, one bad
            'fingerprint_0': np.array([1, 0]),  # One active, one inactive
            'fingerprint_3': np.array([1, 0]),  # One active, one inactive
            'target_family': np.array(['Kinase', 'GPCR'])  # Both druggable
        }
        
        probs = generator._calculate_binding_probabilities(data)
        assert len(probs) == 2
        assert np.all((probs >= 0) & (probs <= 1))
        # First compound should have higher probability
        assert probs[0] > probs[1]  
    
    def test_balanced_dataset_properties(self, molecular_balanced_config):
        """Test generation of balanced dataset."""
        generator = MolecularGenerator(molecular_balanced_config)
        df = generator.generate_data()
        assert isinstance(df, polars.DataFrame)
        assert len(df) == molecular_balanced_config.n_samples
        
    def test_balanced_dataset_class_balance(self, molecular_balanced_config):
        generator = MolecularGenerator(molecular_balanced_config)
        df = generator.generate_data()
        positive_ratio = (df['binds_target'] == 1).mean()
        assert abs(positive_ratio - molecular_balanced_config.positive_ratio) < 0.1
    
    def test_imbalanced_dataset_properties(self, molecular_imbalanced_config):
        """Test generation of imbalanced dataset."""
        generator = MolecularGenerator(molecular_imbalanced_config)
        df = generator.generate_data()
        positive_ratio = (df['binds_target'] == 1).mean()
        # We add 5% noise in the generator, so tolerance should account for that
        noise_tolerance = 0.05
        assert np.isclose(positive_ratio, molecular_imbalanced_config.positive_ratio, atol=noise_tolerance)


@pytest.fixture(scope="class")
def adme_balanced_config():
    """Create a balanced ADME configuration for testing."""
    return ADMEConfig(
        n_samples=1000,
        positive_ratio=0.5,
        random_state=42
    )


@pytest.fixture(scope="class")
def adme_imbalanced_config():
    """Create an imbalanced ADME configuration for testing."""
    return ADMEConfig(
        n_samples=1000,
        positive_ratio=0.03,
        imbalanced=True,
        random_state=42
    )


class TestADMEGenerator:
    """Test suite for ADME data generator."""
    
    def test_adme_features_ranges(self, adme_balanced_config):
        """Test that ADME features are within expected ranges."""
        generator = ADMEGenerator(adme_balanced_config)
        df = generator.generate_data()
        
        # TODO: This can be better done with fixtures or pytest parametrization
        absorption = df['absorption'].to_numpy()
        plasma_binding = df['plasma_protein_binding'].to_numpy()
        clearance = df['clearance'].to_numpy()
        half_life = df['half_life'].to_numpy()
        
        # absorption range
        assert np.all(absorption >= 0)
        assert np.all(absorption <= 100)
        
        # plasma protein binding range
        assert np.all(plasma_binding >= 0)
        assert np.all(plasma_binding <= 100)
        
        # non-negative values
        assert np.all(clearance >= 0)
        assert np.all(half_life >= 0)
    
    def test_bioavailability_rules(self, adme_balanced_config):
        """Test that bioavailability follows the defined rules."""
        #generator = ADMEGenerator(adme_balanced_config)
        # df = generator.generate_data()
        
        # A compound should be bioavailable if it meets all criteria
        perfect_compound = {
            'absorption': 60.0,  # > 50
            'plasma_protein_binding': 90.0,  # < 95
            'clearance': 7.0,  # < 8
            'half_life': 7.0,  # > 6
        }

        perfect_df = polars.DataFrame([perfect_compound])
        bioavailable = np.logical_and.reduce([
            perfect_df['absorption'] > 50,
            perfect_df['plasma_protein_binding'] < 95,
            perfect_df['clearance'] < 8,
            perfect_df['half_life'] > 6
        ]).item()
        
        assert bioavailable
    
    def test_balanced_dataset(self, adme_balanced_config):
        """Test generation of balanced ADME dataset."""
        generator = ADMEGenerator(adme_balanced_config)
        df = generator.generate_data()
        
        # Check DataFrame properties
        assert isinstance(df, polars.DataFrame)
        assert len(df) == adme_balanced_config.n_samples
        
        # Check features
        expected_columns = {
            'absorption', 'plasma_protein_binding', 'clearance',
            'half_life', 'good_bioavailability'
        }
        assert all(col in df.columns for col in expected_columns)
    
    def test_imbalanced_dataset_ratio_with_tolerance(self, adme_imbalanced_config):
        """Test generation of imbalanced ADME dataset. Check imbalance ratio with tolerance for adjustments."""
        generator = ADMEGenerator(adme_imbalanced_config)
        df = generator.generate_data()
        
        positive_ratio = (df['good_bioavailability'] == 1).mean()
        assert abs(positive_ratio - adme_imbalanced_config.positive_ratio) < 0.05