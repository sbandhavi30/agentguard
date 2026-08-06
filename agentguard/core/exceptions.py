class AgentGuardError(Exception):
    """Base class for all AgentGuard errors."""


class CheckpointWriteError(AgentGuardError):
    """Storage backend failed to save checkpoint."""


class RestoreError(AgentGuardError):
    """No checkpoint found or checkpoint bytes corrupted."""


class DeserializationError(AgentGuardError):
    """Adapter failed to deserialize state bytes back to framework state."""


class BackendConnectionError(AgentGuardError):
    """Storage backend unreachable."""
