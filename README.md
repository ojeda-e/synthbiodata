
# synthbiodata

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Polars](https://img.shields.io/badge/powered%20by-Polars-CB4C78.svg)](https://pola.rs/)

A Python package for generating synthetic drug discovery data that mimics real-world scenarios using realistic molecular descriptors and target properties.

> **⚠️ WARNING:** This package generates *synthetic* data for testing, and educational purposes only.  
> The data produced does **not** represent real biological or chemical measurements and should **not** be used for clinical, regulatory, or production applications.


## Features

- Generate synthetic molecular descriptors with realistic ranges (MW, LogP, TPSA, HBD, HBA, etc.)
- Simulate target protein families and their properties (GPCR, Kinase, Protease, etc.)
- Create chemical fingerprints as binary features
- Calculate binding probabilities based on molecular properties
- Generate ADME (Absorption, Distribution, Metabolism, Excretion) data
- Support for both balanced and imbalanced datasets
- Configurable data generation parameters
- Polars DataFrame output for efficient data manipulation

## Installation

```bash
pip install synthbiodata
```

or with my favourite package manager, [uv](https://docs.astral.sh/uv/):

```bash
uv pip install synthbiodata
```

## Quick Start

```python
from synthbiodata import generate_sample_data

# Generate molecular descriptor data with default configuration
df = generate_sample_data(data_type="molecular-descriptors")
print(f"Generated {len(df)} samples with {len(df.columns)} features")

# Generate ADME data
df_adme = generate_sample_data(data_type="adme")
print(f"Generated {len(df_adme)} samples with {len(df_adme.columns)} features")
```

For more control over the data generation process:

```python
from synthbiodata import create_config, generate_sample_data

# Create a custom configuration for molecular descriptors
config = create_config(
    data_type="molecular-descriptors",
    n_samples=1000,
    positive_ratio=0.1,
    imbalanced=True,
    random_state=42
)

# Generate data
df = generate_sample_data(config=config)

# Print results
print(f"Total samples: {len(df)}")
print(f"Features: {len(df.columns) - 1}")  # Exclude target column
print(f"Positive ratio: {df['binds_target'].mean():.1%}")
```

## Data Types

### Molecular Descriptors
Generate synthetic molecular data with features like:
- Molecular weight, LogP, TPSA
- Hydrogen bond donors/acceptors
- Rotatable bonds, aromatic rings
- Chemical fingerprints
- Target protein families (GPCR, Kinase, Protease, etc.)

### ADME Data
Generate ADME (Absorption, Distribution, Metabolism, Excretion) data with:
- Absorption percentages
- Plasma protein binding
- Clearance rates and half-life
- Bioavailability predictions

## Configuration

The configuration system provides extensive options:

- **BaseConfig**: Common parameters (sample size, positive ratio, train/test splits)
- **MolecularConfig**: Molecular descriptor ranges and target family settings
- **ADMEConfig**: ADME-specific parameters (absorption, clearance, etc.)
- Support for both balanced and imbalanced datasets
- Customizable random seeds for reproducibility


