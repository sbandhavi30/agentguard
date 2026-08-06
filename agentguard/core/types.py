from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CheckpointMeta:
    run_id: str
    step: int
    trigger: str
    timestamp: datetime
    framework: str
    token_count: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class RestoredState:
    step: int
    state: bytes
