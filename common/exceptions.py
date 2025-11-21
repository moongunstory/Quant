# common/exceptions.py
"""
Custom exception hierarchy for the Quant pipeline.
Provides clear error categorization and handling strategies.
"""


class QuantPipelineError(Exception):
    """Base exception for all Quant pipeline errors."""
    pass


class DataCollectionError(QuantPipelineError):
    """
    Non-critical error during data collection.
    Pipeline can continue with existing data.

    Examples:
    - API rate limit exceeded
    - Temporary network failure
    - Single source unavailable
    """
    pass


class CriticalDataError(QuantPipelineError):
    """
    Critical error requiring pipeline termination.

    Examples:
    - Essential data file missing
    - Database corruption
    - Invalid configuration
    """
    pass


class FeatureEngineeringError(QuantPipelineError):
    """
    Error during feature computation.

    Examples:
    - Insufficient data for rolling calculation
    - Invalid feature configuration
    - Data type mismatch
    """
    pass


class ModelTrainingError(QuantPipelineError):
    """
    Error during model training.

    Examples:
    - Insufficient training samples
    - All labels are same class
    - Model convergence failure
    """
    pass


class HPOError(QuantPipelineError):
    """
    Error during hyperparameter optimization.

    Examples:
    - Trial evaluation failure
    - Invalid search space
    - Insufficient trials completed
    """
    pass
