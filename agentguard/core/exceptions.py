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


class AgentLoopDetectedError(AgentGuardError):
    """Agent exceeded max_steps — likely stuck in a tool-call loop."""
    def __init__(self, run_id: str, step: int):
        self.run_id = run_id
        self.step = step
        super().__init__(f"Agent loop detected: run_id={run_id!r} exceeded max_steps at step {step}")
