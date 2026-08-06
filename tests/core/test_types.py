from datetime import datetime, timezone
from agentguard.core.types import CheckpointMeta, RestoredState


def test_checkpoint_meta_frozen():
    meta = CheckpointMeta(
        run_id="run-1", step=5, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework="langgraph",
    )
    try:
        meta.step = 6  # type: ignore
        assert False, "should be immutable"
    except Exception:
        pass


def test_checkpoint_meta_defaults():
    meta = CheckpointMeta(
        run_id="r", step=1, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework="anthropic",
    )
    assert meta.token_count == 0
    assert meta.cost_usd == 0.0


def test_restored_state_frozen():
    rs = RestoredState(step=3, state=b"hello")
    try:
        rs.step = 4  # type: ignore
        assert False, "should be immutable"
    except Exception:
        pass


def test_restored_state_holds_bytes():
    payload = b"\x00\x01\x02\xff"
    rs = RestoredState(step=1, state=payload)
    assert rs.state == payload
