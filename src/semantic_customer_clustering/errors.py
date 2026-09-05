"""Domain-specific exceptions with actionable messages."""


class ClusteringError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(ClusteringError):
    """Configuration is missing or invalid."""


class DataValidationError(ClusteringError):
    """Input data does not satisfy the clustering contract."""


class DependencyUnavailableError(ClusteringError):
    """An optional feature's dependency is not installed."""


class ModelExecutionError(ClusteringError):
    """A model or external adapter returned unusable output."""
