"""Tests for configuration module."""

import pytest
from synthbiodata.config import (
    BaseConfig,
    MolecularConfig,
    ADMEConfig,
    create_config,
    generate_sample_data,
)


def test_base_config_defaults():
    """Test that BaseConfig has correct default values."""
    config = BaseConfig()
    assert config.n_samples == 10000
    assert config.positive_ratio == 0.03
    assert config.test_size == 0.2
    assert config.val_size == 0.2
    assert config.random_state == 42
    assert not config.imbalanced


def test_base_config_validation_valid_splits():
    """Test BaseConfig valid splits."""
    config = BaseConfig(test_size=0.2, val_size=0.2)
    assert config.test_size + config.val_size < 1.0


def test_base_config_validation_invalid_splits():
    """Test BaseConfig invalid splits."""
    with pytest.raises(ValueError, match="Sum of test_size and val_size must be less than 1"):
        BaseConfig(test_size=0.5, val_size=0.5)

def test_base_config_validation_invalid_positive_ratio():
    """Test BaseConfig invalid positive ratio."""
    with pytest.raises(ValueError, match="positive_ratio must be between 0 and 1"):
        BaseConfig(positive_ratio=1.5)


def test_molecular_config_defaults():
    """Test MolecularConfig default values."""
    config = MolecularConfig()
    
    # Check molecular weight parameters
    assert config.mw_mean == 350.0
    assert config.mw_std == 100.0
    assert config.mw_min == 150.0
    assert config.mw_max == 600.0
    
    # Check target families
    assert len(config.target_families) == 5
    assert len(config.target_family_probs) == 5
    assert abs(sum(config.target_family_probs) - 1.0) < 1e-6


def test_molecular_config_validation():
    """Test MolecularConfig validation rules."""
    # Test invalid molecular weight range
    with pytest.raises(ValueError, match="mw_min must be less than mw_max"):
        MolecularConfig(mw_min=400.0, mw_max=300.0)
    
    # Test invalid target family probabilities
    with pytest.raises(ValueError, match="target_family_probs must sum to 1.0"):
        MolecularConfig(
            target_families=['A', 'B'],
            target_family_probs=[0.8, 0.8]
        )
    
    # Test mismatched lengths
    with pytest.raises(ValueError, match="Length of target_families must match target_family_probs"):
        MolecularConfig(
            target_families=['A', 'B'],
            target_family_probs=[1.0]
        )


def test_adme_config_defaults():
    """Test ADMEConfig default values."""
    config = ADMEConfig()
    
    # absorption params
    assert config.absorption_mean == 70.0
    assert config.absorption_std == 20.0
    
    # metabolism params
    assert config.clearance_mean == 5.0
    assert config.half_life_mean == 12.0


def test_adme_config_invalid_params():
    """Test ADMEConfig invalid params."""
    # invalid absorption mean
    with pytest.raises(ValueError, match="absorption_mean must be between 0 and 100"):
        ADMEConfig(absorption_mean=150.0)
    
    # invalid clearance
    with pytest.raises(ValueError, match="clearance_mean must be positive"):
        ADMEConfig(clearance_mean=-1.0)


def test_create_config_molecular_descriptors():
    """Test config factory function for molecular descriptors."""
    mol_config = create_config("molecular-descriptors")
    assert isinstance(mol_config, MolecularConfig)
    
def test_create_config_adme():
    """Test config factory function for adme."""
    adme_config = create_config("adme")
    assert isinstance(adme_config, ADMEConfig)


def test_create_config_adme_imbalanced():   
    """Test config factory function for adme imbalanced."""
    imbal_config = create_config("molecular-descriptors", imbalanced=True)
    assert imbal_config.imbalanced
    assert imbal_config.positive_ratio == 0.03
    
def test_create_config_molecular_descriptors_custom():
    custom_config: MolecularConfig = create_config(
        "molecular-descriptors",
        n_samples=5000,
        mw_mean=400.0,
        target_families=['A', 'B'],
        target_family_probs=[0.6, 0.4]
    )
    assert isinstance(custom_config, MolecularConfig)  # Type assertion
    assert custom_config.n_samples == 5000
    assert custom_config.mw_mean == 400.0
    assert custom_config.target_families == ['A', 'B']
    assert custom_config.target_family_probs == [0.6, 0.4]


def test_config_unimplemented_data_types():
    """Test that unimplemented data types raise appropriate errors."""
    with pytest.raises(ValueError, match="'something-unexistent' is not a valid DataType"):
        create_config(data_type="something-unexistent")
    
def test_generate_sample_data_unimplemented_data_types():
    """Test that generate_sample_data raises error for unimplemented types."""
    with pytest.raises(ValueError, match="'something-unexistent' is not a valid DataType"):
        generate_sample_data(data_type="something-unexistent")


