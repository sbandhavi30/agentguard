from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CheckpointMeta:
    pass


@dataclass(frozen=True)
class RestoredState:
    pass
