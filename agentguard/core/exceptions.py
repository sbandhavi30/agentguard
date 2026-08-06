class AgentGuardError(Exception):
    pass


class CheckpointWriteError(AgentGuardError):
    pass


class RestoreError(AgentGuardError):
    pass


class DeserializationError(AgentGuardError):
    pass


class BackendConnectionError(AgentGuardError):
    pass
