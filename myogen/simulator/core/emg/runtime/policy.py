"""Public execution options; scientific parameters belong to the model."""
from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class ExecutionPolicy:
    backend: str = "cpu"
    memory_budget_mib: float | None = None
    reserve_fraction: float = 0.20
    reserve_min_mib: float = 4096
    max_workers: int = 2
    cache_dir: str | Path | None = None

    def __post_init__(self):
        if self.backend != "cpu":
            raise ValueError("Optimized execution supports backend='cpu' only")
        if not isinstance(self.max_workers, int) or isinstance(self.max_workers, bool) or self.max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if not math.isfinite(self.reserve_fraction) or not 0 <= self.reserve_fraction < 1:
            raise ValueError("reserve_fraction must be finite and in [0, 1)")
        if not math.isfinite(self.reserve_min_mib) or self.reserve_min_mib < 0:
            raise ValueError("reserve_min_mib must be finite and nonnegative")
        if self.memory_budget_mib is not None:
            if not math.isfinite(self.memory_budget_mib) or self.memory_budget_mib <= 0:
                raise ValueError("memory_budget_mib must be finite and positive")
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
