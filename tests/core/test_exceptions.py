from agentguard.core.exceptions import (
    AgentGuardError,
    CheckpointWriteError,
    RestoreError,
    DeserializationError,
    BackendConnectionError,
)


def test_all_inherit_base():
    for cls in [CheckpointWriteError, RestoreError, DeserializationError, BackendConnectionError]:
        assert issubclass(cls, AgentGuardError)


def test_raise_restore_error():
    try:
        raise RestoreError("no checkpoint found for run_id=xyz")
    except AgentGuardError as e:
        assert "xyz" in str(e)
