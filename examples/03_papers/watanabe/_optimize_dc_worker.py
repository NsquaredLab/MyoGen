"""Internal worker: run N Optuna trials against the shared study. Not a gallery example."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import optuna  # noqa: E402

from _oscillating_dc_helpers import (  # noqa: E402
    STUDY_NAME,
    TIMEOUT_SECONDS,
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
    # Seed this worker's TPE sampler explicitly. Optuna does NOT persist samplers
    # to storage, so a bare load_study() would build a fresh *unseeded* sampler in
    # every worker, making the dc_offset suggestions non-reproducible despite the
    # study being created with a seed. (Bit-reproducibility across parallel workers
    # sharing one study is still not achievable — trial interleaving is
    # nondeterministic — but this removes the unseeded-sampler source of drift.)
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=make_storage(),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    # Keep the wall-clock guard that existed before the parallelization: each
    # worker stops at n_trials OR TIMEOUT_SECONDS, whichever comes first (workers
    # run concurrently, so this bounds the whole optimization to ~TIMEOUT_SECONDS).
    study.optimize(objective, n_trials=args.n_trials, timeout=TIMEOUT_SECONDS)


if __name__ == "__main__":
    main()
