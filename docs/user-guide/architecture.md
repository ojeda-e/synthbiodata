# System Architecture

SynthBioData is built on a modular, extensible architecture that separates concerns and provides a clean interface for generating different types of synthetic biological data. This design makes the system maintainable, testable, and easy to extend with new data types.

## Core Design Principles

The architecture follows several key design principles:

**Separation of Concerns**: Configuration, data generation, and business logic are separated into distinct modules.

**Type Safety**: Full type hints and Pydantic validation ensure robust parameter handling and clear interfaces.

**Extensibility**: New data types can be added by implementing the base generator interface without modifying existing code.

**Reproducibility**: All generators use seeded random number generation to ensure consistent results.

**Performance**: Built on Polars for efficient data manipulation and NumPy for fast numerical operations.

## Component Overview

### Configuration Layer

The configuration system provides type-safe parameter management:

- **`BaseConfig`**: Common parameters shared by all data types (sample size, random state, validation splits)
- **`MolecularConfig`**: Molecular-specific parameters (MW, LogP, TPSA ranges, target families)
- **`ADMEConfig`**: ADME-specific parameters (absorption, clearance, half-life distributions)

All configurations use Pydantic for automatic validation and provide clear error messages for invalid parameters.

### Generator Layer

The generator layer implements the actual data generation logic:

- **`BaseGenerator`**: Abstract base class providing common functionality
- **`MolecularGenerator`**: Generates molecular descriptor data with target binding probabilities
- **`ADMEGenerator`**: Generates ADME data with bioavailability classifications

Each generator is self-contained and responsible for its specific data type.

### Factory Layer

The factory layer provides convenient interfaces for creating configurations and generators:

- **`create_config()`**: Type-safe configuration creation with automatic parameter validation
- **`create_generator()`**: Generator instantiation based on configuration type
- **`generate_sample_data()`**: High-level function for quick data generation

## Inheritance Hierarchy

The data generation system follows a clean inheritance hierarchy:

```
BaseGenerator (Abstract)
├── MolecularGenerator
└── ADMEGenerator
```

**BaseGenerator** is responsible for storing and validating configuration settings, ensuring reproducible results through seeded random number generation (using NumPy) and deterministic fake data creation (using Faker). It also defines the abstract interface that all data generators must implement.

**MolecularGenerator** extends BaseGenerator by providing molecular descriptor generation (including MW, LogP, TPSA, and related properties), chemical fingerprint generation, target protein family simulation, and binding probability calculation.

**ADMEGenerator** extends BaseGenerator by providing ADME property generation (including absorption, distribution, metabolism, and excretion) as well as bioavailability classification and pharmacokinetic parameter simulation.

## Data Flow

The typical data generation flow follows these steps:

```mermaid
graph LR
    A[Configuration] --> B[Validation]
    B --> C[Generator]
    C --> D[Data Generation]
    D --> E[DataFrame]
    E --> F[Result]
    
    style A fill:#e1f5fe
    style F fill:#e8f5e8
```

### Process Steps

1. **Configuration Creation**: User creates or loads a configuration object
2. **Validation**: Pydantic validates all parameters and raises errors for invalid values
3. **Generator Instantiation**: Factory creates appropriate generator based on configuration type
4. **Data Generation**: Generator creates synthetic data using statistical distributions
5. **Post-processing**: Data is formatted into Polars DataFrames with proper column names and types
6. **Return**: Complete dataset is returned to the user





## Extensibility

The architecture makes it easy to add new data types:

1. **Create Configuration**: Extend `BaseConfig` with new parameters
2. **Implement Generator**: Create a new generator class inheriting from `BaseGenerator`
3. **Update Factory**: Add the new type to factory functions
4. **Add Constants**: Define default values in constants module

This design ensures that new features can be added without breaking existing functionality or requiring changes to the core architecture.

## Error Handling

The system provides comprehensive error handling at multiple levels:

- **Configuration Validation**: Pydantic catches invalid parameters before data generation
- **Range Validation**: Custom validators ensure parameters are within realistic biological ranges
- **Type Safety**: Type hints prevent many common programming errors
- **Clear Error Messages**: Detailed error messages help users understand and fix configuration issues

## Performance Considerations

The architecture is designed for performance:

- **Efficient Data Structures**: Polars DataFrames for fast data manipulation
- **Vectorized Operations**: NumPy for efficient numerical computations
- **Memory Management**: Generators create data in chunks when possible
- **Lazy Evaluation**: Polars supports lazy evaluation for large datasets

This design ensures that SynthBioData can handle both small experimental datasets and large-scale data generation tasks efficiently.

