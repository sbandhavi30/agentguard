import fnmatch
from dataclasses import dataclass


@dataclass(frozen=True)
class TriggerMeta:
    tool_name: str | None = None
    token_count: int = 0
    total_budget: int = 0
    is_destructive: bool = False
    step: int = 0


class TriggerPolicy:
    def __init__(
        self,
        destructive_tools: list[str] | None = None,
        token_pressure_threshold: float = 0.80,
    ) -> None:
        self.destructive_tools = destructive_tools or []
        self.token_pressure_threshold = token_pressure_threshold

    def should_checkpoint(self, meta: TriggerMeta) -> tuple[bool, str]:
        if meta.is_destructive:
            return True, "destructive_action"

        if meta.tool_name:
            for pattern in self.destructive_tools:
                if fnmatch.fnmatch(meta.tool_name, pattern):
                    return True, "destructive_action"

        if meta.total_budget > 0:
            if meta.token_count / meta.total_budget >= self.token_pressure_threshold:
                return True, "token_pressure"

        return True, "tool_call"
