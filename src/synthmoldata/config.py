"""
Configuration module for synthetic biological data generation.
"""

from enum import Enum
from typing import Optional, overload, Literal
from pydantic import BaseModel, Field, model_validator
import polars

class DataType(str, Enum):
    """Supported data types for generation."""
    MOLECULAR = "molecular-descriptors"
    ADME = "adme"
    # TODO: Add other data types as needed (cancer, dose-response, etc.)

class BaseConfig(BaseModel):
    """Base configuration for all data types."""
    n_samples: int = Field(10000, description="Number of samples to generate")
    positive_ratio: float = Field(0.03, description="Ratio of positive samples")
    test_size: float = Field(0.2, description="Test set size ratio")
    val_size: float = Field(0.2, description="Validation set size ratio")
    random_state: int = Field(42, description="Random seed for reproducibility")
    imbalanced: bool = Field(False, description="Whether to generate imbalanced dataset")

    @model_validator(mode='after')
    def validate_splits(self) -> 'BaseConfig':
        """Validate dataset split ratios."""
        if self.test_size + self.val_size >= 1:
            raise ValueError("Sum of test_size and val_size must be less than 1")
        if self.positive_ratio <= 0 or self.positive_ratio >= 1:
            raise ValueError("positive_ratio must be between 0 and 1")
        return self


class MolecularConfig(BaseConfig):
    """Configuration for molecular descriptor data."""
    # Molecular descriptor ranges
    mw_mean: float = Field(350.0, description="Mean molecular weight")
    mw_std: float = Field(100.0, description="Standard deviation of molecular weight")
    mw_min: float = Field(150.0, description="Minimum molecular weight")
    mw_max: float = Field(600.0, description="Maximum molecular weight")
    
    logp_mean: float = Field(2.5, description="Mean LogP value")
    logp_std: float = Field(1.5, description="Standard deviation of LogP")
    logp_min: float = Field(-2.0, description="Minimum LogP value")
    logp_max: float = Field(6.0, description="Maximum LogP value")
    
    tpsa_mean: float = Field(80.0, description="Mean TPSA value")
    tpsa_std: float = Field(40.0, description="Standard deviation of TPSA")
    tpsa_min: float = Field(0.0, description="Minimum TPSA value")
    tpsa_max: float = Field(200.0, description="Maximum TPSA value")
    
    # Target protein families
    target_families: list[str] = Field(
        default=['GPCR', 'Kinase', 'Protease', 'Nuclear Receptor', 'Ion Channel'],
        description="List of target protein families"
    )
    target_family_probs: list[float] = Field(
        default=[0.3, 0.25, 0.2, 0.15, 0.1],
        description="Probability distribution for target families"
    )

    @model_validator(mode='after')
    def validate_ranges(self) -> 'MolecularConfig':
        """Validate molecular descriptor ranges."""
        if self.mw_min >= self.mw_max:
            raise ValueError("mw_min must be less than mw_max")
        if self.logp_min >= self.logp_max:
            raise ValueError("logp_min must be less than logp_max")
        if self.tpsa_min >= self.tpsa_max:
            raise ValueError("tpsa_min must be less than tpsa_max")
        if len(self.target_families) != len(self.target_family_probs):
            raise ValueError("Length of target_families must match target_family_probs")
        if abs(sum(self.target_family_probs) - 1.0) > 1e-6:
            raise ValueError("target_family_probs must sum to 1.0")
        return self


class ADMEConfig(BaseConfig):
    """Configuration for ADME data generation."""
    # Absorption parameters
    absorption_mean: float = Field(70.0, description="Mean absorption percentage")
    absorption_std: float = Field(20.0, description="Standard deviation of absorption")
    
    # Distribution parameters
    plasma_protein_binding_mean: float = Field(85.0, description="Mean plasma protein binding percentage")
    plasma_protein_binding_std: float = Field(15.0, description="Standard deviation of plasma protein binding")
    
    # Metabolism parameters
    clearance_mean: float = Field(5.0, description="Mean clearance rate (L/h)")
    clearance_std: float = Field(2.0, description="Standard deviation of clearance")
    half_life_mean: float = Field(12.0, description="Mean half-life (hours)")
    half_life_std: float = Field(6.0, description="Standard deviation of half-life")
    
    # Excretion parameters
    renal_clearance_ratio: float = Field(0.3, description="Ratio of renal to total clearance")

    @model_validator(mode='after')
    def validate_parameters(self) -> 'ADMEConfig':
        """Validate ADME parameters."""
        if self.absorption_mean < 0 or self.absorption_mean > 100:
            raise ValueError("absorption_mean must be between 0 and 100")
        if self.plasma_protein_binding_mean < 0 or self.plasma_protein_binding_mean > 100:
            raise ValueError("plasma_protein_binding_mean must be between 0 and 100")
        if self.clearance_mean <= 0:
            raise ValueError("clearance_mean must be positive")
        if self.half_life_mean <= 0:
            raise ValueError("half_life_mean must be positive")
        if self.renal_clearance_ratio < 0 or self.renal_clearance_ratio > 1:
            raise ValueError("renal_clearance_ratio must be between 0 and 1")
        return self


def create_config(data_type: str, imbalanced: bool = False, **kwargs) -> BaseConfig:
    """Create a configuration for the specified data type."""
    config_classes = {
        DataType.MOLECULAR: MolecularConfig,
        DataType.ADME: ADMEConfig,
        # Add other config classes as needed
    }
    
    data_type_enum = DataType(data_type)
    config_class = config_classes.get(data_type_enum, BaseConfig)
    
    # Set imbalanced flag and update kwargs
    kwargs['imbalanced'] = imbalanced
    if imbalanced and 'positive_ratio' not in kwargs:
        kwargs['positive_ratio'] = 0.03  # Default imbalanced ratio
        
    return config_class(**kwargs)


def generate_sample_data(
    data_type: str = "molecular-descriptors",
    imbalanced: bool = False,
    config: Optional[BaseConfig] = None,
    **kwargs
) -> "polars.DataFrame":
    """Generate synthetic biological data."""
    from .core.generator import ADMEGenerator, MolecularGenerator
    
    if config is None:
        # Set default values only if not provided in kwargs
        if 'n_samples' not in kwargs:
            kwargs['n_samples'] = 5000
        if 'positive_ratio' not in kwargs:
            kwargs['positive_ratio'] = 0.03 if imbalanced else 0.5
        if 'random_state' not in kwargs:
            kwargs['random_state'] = 42
            
        config = create_config(data_type, imbalanced=imbalanced, **kwargs)
    
    # Use the config's type to determine the generator
    if isinstance(config, MolecularConfig):
        generator_class = MolecularGenerator
    elif isinstance(config, ADMEConfig):
        generator_class = ADMEGenerator
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")
    
    generator = generator_class(config)
    return generator.generate_data()