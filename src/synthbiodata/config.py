"""
Configuration module for synthetic biological data generation.
"""

from enum import Enum
from typing import Optional, overload, Literal
from pydantic import BaseModel, Field, model_validator
import polars

from .constants import DATASET_DEFAULTS, MOLECULAR_DEFAULTS, ADME_DEFAULTS, TARGET_FAMILIES, TARGET_FAMILY_PROBS
from .exceptions import RangeError, DistributionError, DataTypeError
from .logging import logger

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
        total_split = self.test_size + self.val_size
        if total_split >= 1:
            logger.error(f"Invalid split ratios: test_size={self.test_size}, val_size={self.val_size}, total={total_split}")
            raise RangeError("total split ratio", total_split, max_val=1.0)
            
        if self.positive_ratio <= 0 or self.positive_ratio >= 1:
            logger.error(f"Invalid positive ratio: {self.positive_ratio}")
            raise RangeError("positive_ratio", self.positive_ratio, min_val=0.0, max_val=1.0)
            
        logger.debug("Validated dataset split ratios successfully")
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
        # Validate min/max ranges
        for param in ['mw', 'logp', 'tpsa']:
            min_val = getattr(self, f"{param}_min")
            max_val = getattr(self, f"{param}_max")
            if min_val >= max_val:
                logger.error(f"Invalid {param} range: min={min_val}, max={max_val}")
                raise RangeError(f"{param}_min", min_val, max_val=max_val)
                
        # Validate standard deviations
        for param in ['mw', 'logp', 'tpsa']:
            std_val = getattr(self, f"{param}_std")
            if std_val <= 0:
                logger.error(f"Invalid {param} standard deviation: {std_val}")
                raise RangeError(f"{param}_std", std_val, min_val=0)

        # Validate target distributions
        if len(self.target_families) != len(self.target_family_probs):
            logger.error(
                f"Mismatched lengths: target_families={len(self.target_families)}, "
                f"target_family_probs={len(self.target_family_probs)}"
            )
            raise DistributionError(
                f"Length mismatch: target_families ({len(self.target_families)}) "
                f"!= target_family_probs ({len(self.target_family_probs)})"
            )
            
        prob_sum = sum(self.target_family_probs)
        if abs(prob_sum - 1.0) > 1e-6:
            logger.error(f"Target family probabilities sum to {prob_sum}, should be 1.0")
            raise DistributionError(
                f"Target family probabilities must sum to 1.0, got {prob_sum}"
            )
            
        logger.debug("Validated molecular descriptor ranges successfully")
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
        # Validate means and ratios
        if self.absorption_mean < 0 or self.absorption_mean > 100:
            logger.error(f"Invalid absorption mean: {self.absorption_mean}")
            raise RangeError("absorption_mean", self.absorption_mean, min_val=0, max_val=100)
            
        if self.plasma_protein_binding_mean < 0 or self.plasma_protein_binding_mean > 100:
            logger.error(f"Invalid plasma protein binding mean: {self.plasma_protein_binding_mean}")
            raise RangeError("plasma_protein_binding_mean", self.plasma_protein_binding_mean, min_val=0, max_val=100)
            
        if self.clearance_mean <= 0:
            logger.error(f"Invalid clearance mean: {self.clearance_mean}")
            raise RangeError("clearance_mean", self.clearance_mean, min_val=0)
            
        if self.half_life_mean <= 0:
            logger.error(f"Invalid half life mean: {self.half_life_mean}")
            raise RangeError("half_life_mean", self.half_life_mean, min_val=0)
            
        if self.renal_clearance_ratio < 0 or self.renal_clearance_ratio > 1:
            logger.error(f"Invalid renal clearance ratio: {self.renal_clearance_ratio}")
            raise RangeError("renal_clearance_ratio", self.renal_clearance_ratio, min_val=0, max_val=1)

        # Validate standard deviations
        if self.absorption_std <= 0:
            logger.error(f"Invalid absorption std: {self.absorption_std}")
            raise RangeError("absorption_std", self.absorption_std, min_val=0)
            
        if self.plasma_protein_binding_std <= 0:
            logger.error(f"Invalid plasma protein binding std: {self.plasma_protein_binding_std}")
            raise RangeError("plasma_protein_binding_std", self.plasma_protein_binding_std, min_val=0)
            
        if self.clearance_std <= 0:
            logger.error(f"Invalid clearance std: {self.clearance_std}")
            raise RangeError("clearance_std", self.clearance_std, min_val=0)
            
        if self.half_life_std <= 0:
            logger.error(f"Invalid half life std: {self.half_life_std}")
            raise RangeError("half_life_std", self.half_life_std, min_val=0)
            
        logger.debug("Validated ADME parameters successfully")
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
    
    try:
        data_type_enum = DataType(data_type)
    except ValueError:
        logger.error(f"Invalid data type: {data_type}")
        raise DataTypeError(f"'{data_type}' is not a valid DataType")
        
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