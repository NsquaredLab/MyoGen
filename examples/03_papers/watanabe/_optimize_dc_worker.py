"""Internal worker: run N Optuna trials against the shared study. Not a gallery example."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import optuna  # noqa: E402

from _oscillating_dc_helpers import (  # noqa: E402
    STUDY_NAME,
    make_storage,
    objective,
)
from myogen import set_random_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Decorrelate per-worker membrane/drive noise; thresholds were already built at
    # import under seed 42 so they stay identical across workers.
    set_random_seed(args.seed)
    study = optuna.load_study(study_name=STUDY_NAME, storage=make_storage())
    study.optimize(objective, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
