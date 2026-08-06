import pytest
from agentguard.core.triggers import TriggerMeta, TriggerPolicy


def make_meta(**kwargs):
    defaults = dict(
        tool_name=None, token_count=0, total_budget=0,
        is_destructive=False, step=0,
    )
    defaults.update(kwargs)
    return TriggerMeta(**defaults)


def test_explicit_destructive_flag():
    policy = TriggerPolicy()
    result, reason = policy.should_checkpoint(make_meta(is_destructive=True))
    assert result is True
    assert reason == "destructive_action"


def test_wildcard_destructive_match():
    policy = TriggerPolicy(destructive_tools=["delete_*", "drop_*"])
    result, reason = policy.should_checkpoint(make_meta(tool_name="delete_bucket"))
    assert result is True
    assert reason == "destructive_action"


def test_wildcard_no_match_falls_through():
    policy = TriggerPolicy(destructive_tools=["delete_*"])
    result, reason = policy.should_checkpoint(make_meta(tool_name="list_buckets"))
    assert result is True
    assert reason == "tool_call"


def test_token_pressure_triggers():
    policy = TriggerPolicy(token_pressure_threshold=0.80)
    result, reason = policy.should_checkpoint(
        make_meta(token_count=8500, total_budget=10000)
    )
    assert result is True
    assert reason == "token_pressure"


def test_token_pressure_below_threshold():
    policy = TriggerPolicy(token_pressure_threshold=0.80)
    result, reason = policy.should_checkpoint(
        make_meta(token_count=7000, total_budget=10000)
    )
    assert result is True
    assert reason == "tool_call"


def test_zero_budget_no_division_error():
    policy = TriggerPolicy(token_pressure_threshold=0.80)
    result, reason = policy.should_checkpoint(make_meta(total_budget=0, token_count=9999))
    assert result is True
    assert reason == "tool_call"


def test_destructive_flag_takes_priority_over_token_pressure():
    policy = TriggerPolicy(token_pressure_threshold=0.10)
    result, reason = policy.should_checkpoint(
        make_meta(is_destructive=True, token_count=9999, total_budget=10000)
    )
    assert reason == "destructive_action"


def test_trigger_meta_frozen():
    meta = make_meta(step=1)
    try:
        meta.step = 2  # type: ignore
        assert False, "should be frozen"
    except Exception:
        pass
