"""
Configuration module for synthetic biological data generation.
"""

from enum import Enum
from typing import Optional, overload, Literal
from pydantic import BaseModel, Field, model_validator
import polars

from .constants import DATASET_DEFAULTS, MOLECULAR_DEFAULTS, ADME_DEFAULTS, TARGET_FAMILIES, TARGET_FAMILY_PROBS

class DataType(str, Enum):
    """Supported data types for generation."""
    MOLECULAR = "molecular-descriptors"
    ADME = "adme"
    # TODO: Add other data types as needed (cancer, dose-response, etc.)

class BaseConfig(BaseModel):
    """Base configuration for all data types."""
    n_samples: int = Field(DATASET_DEFAULTS["DEFAULT_SAMPLES"], 
                          description="Number of samples to generate")
    positive_ratio: float = Field(DATASET_DEFAULTS["IMBALANCED_RATIO"], 
                                description="Ratio of positive samples")
    test_size: float = Field(DATASET_DEFAULTS["TEST_SIZE"], 
                           description="Test set size ratio")
    val_size: float = Field(DATASET_DEFAULTS["VAL_SIZE"], 
                          description="Validation set size ratio")
    random_state: int = Field(DATASET_DEFAULTS["RANDOM_SEED"], 
                            description="Random seed for reproducibility")
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
    mw_mean: float = Field(MOLECULAR_DEFAULTS["MW_MEAN"], 
                          description="Mean molecular weight")
    mw_std: float = Field(MOLECULAR_DEFAULTS["MW_STD"], 
                         description="Standard deviation of molecular weight")
    mw_min: float = Field(MOLECULAR_DEFAULTS["MW_MIN"], 
                         description="Minimum molecular weight")
    mw_max: float = Field(MOLECULAR_DEFAULTS["MW_MAX"], 
                         description="Maximum molecular weight")
    
    logp_mean: float = Field(MOLECULAR_DEFAULTS["LOGP_MEAN"], 
                            description="Mean LogP value")
    logp_std: float = Field(MOLECULAR_DEFAULTS["LOGP_STD"], 
                           description="Standard deviation of LogP")
    logp_min: float = Field(MOLECULAR_DEFAULTS["LOGP_MIN"], 
                           description="Minimum LogP value")
    logp_max: float = Field(MOLECULAR_DEFAULTS["LOGP_MAX"], 
                           description="Maximum LogP value")
    
    tpsa_mean: float = Field(MOLECULAR_DEFAULTS["TPSA_MEAN"], 
                            description="Mean TPSA value")
    tpsa_std: float = Field(MOLECULAR_DEFAULTS["TPSA_STD"], 
                           description="Standard deviation of TPSA")
    tpsa_min: float = Field(MOLECULAR_DEFAULTS["TPSA_MIN"], 
                           description="Minimum TPSA value")
    tpsa_max: float = Field(MOLECULAR_DEFAULTS["TPSA_MAX"], 
                           description="Maximum TPSA value")
    
    # Target protein families
    target_families: list[str] = Field(
        default=TARGET_FAMILIES,
        description="List of target protein families"
    )
    target_family_probs: list[float] = Field(
        default=TARGET_FAMILY_PROBS,
        description="Probability distribution for target families"
    )

    @model_validator(mode='after')
    def validate_ranges(self) -> 'MolecularConfig':
        """Validate molecular descriptor ranges and standard deviations."""
        if self.mw_min >= self.mw_max:
            raise ValueError("mw_min must be less than mw_max")
        if self.logp_min >= self.logp_max:
            raise ValueError("logp_min must be less than logp_max")
        if self.tpsa_min >= self.tpsa_max:
            raise ValueError("tpsa_min must be less than tpsa_max")

        if self.mw_std <= 0:
            raise ValueError("mw_std must be positive")
        if self.logp_std <= 0:
            raise ValueError("logp_std must be positive")
        if self.tpsa_std <= 0:
            raise ValueError("tpsa_std must be positive")

        if len(self.target_families) != len(self.target_family_probs):
            raise ValueError("Length of target_families must match target_family_probs")
        if abs(sum(self.target_family_probs) - 1.0) > 1e-6:
            raise ValueError("target_family_probs must sum to 1.0")
        return self


class ADMEConfig(BaseConfig):
    """Configuration for ADME data generation."""
    # Absorption
    absorption_mean: float = Field(ADME_DEFAULTS["ABSORPTION_MEAN"], 
                                 description="Mean absorption percentage")
    absorption_std: float = Field(ADME_DEFAULTS["ABSORPTION_STD"], 
                                description="Standard deviation of absorption")
    
    # Distribution
    plasma_protein_binding_mean: float = Field(ADME_DEFAULTS["PROTEIN_BINDING_MEAN"], 
                                             description="Mean plasma protein binding percentage")
    plasma_protein_binding_std: float = Field(ADME_DEFAULTS["PROTEIN_BINDING_STD"], 
                                            description="Standard deviation of plasma protein binding")
    
    # Metabolism
    clearance_mean: float = Field(ADME_DEFAULTS["CLEARANCE_MEAN"], 
                                description="Mean clearance rate (L/h)")
    clearance_std: float = Field(ADME_DEFAULTS["CLEARANCE_STD"], 
                               description="Standard deviation of clearance")
    half_life_mean: float = Field(ADME_DEFAULTS["HALF_LIFE_MEAN"], 
                                description="Mean half-life (hours)")
    half_life_std: float = Field(ADME_DEFAULTS["HALF_LIFE_STD"], 
                               description="Standard deviation of half-life")
    
    # Excretion
    renal_clearance_ratio: float = Field(ADME_DEFAULTS["RENAL_CLEARANCE_RATIO"], 
                                       description="Ratio of renal to total clearance")

    @model_validator(mode='after')
    def validate_parameters(self) -> 'ADMEConfig':
        """Validate ADME parameters and standard deviations."""
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

        if self.absorption_std <= 0:
            raise ValueError("absorption_std must be positive")
        if self.plasma_protein_binding_std <= 0:
            raise ValueError("plasma_protein_binding_std must be positive")
        if self.clearance_std <= 0:
            raise ValueError("clearance_std must be positive")
        if self.half_life_std <= 0:
            raise ValueError("half_life_std must be positive")
        return self

# Overloads for the create_config function - type hints
@overload
def create_config(data_type: Literal["molecular-descriptors"], imbalanced: bool = False, **kwargs) -> MolecularConfig:
    ...

@overload
def create_config(data_type: Literal["adme"], imbalanced: bool = False, **kwargs) -> ADMEConfig:
    ...

@overload
def create_config(data_type: str, imbalanced: bool = False, **kwargs) -> BaseConfig:
    ...
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