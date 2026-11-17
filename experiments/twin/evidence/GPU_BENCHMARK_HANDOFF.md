# GPU benchmark and gradient qualification: handoff

Evidence for draft Figure "Gradient qualification and computational scaling"
(provisional name; the existing Figure 2 PIC-envelope file is untouched).
All numbers below are parsed from the files in `gpu_benchmark_run_20261117/`
and the three timestamped `gpu_benchmark_*.json` files next to this note.

## Verdicts first

1. **The GPU never ran silently on CPU.** Every GPU row reports
   `jax_platform = gpu` (`CudaDevice(id=0)`, NVIDIA GeForce RTX 4090); every CPU row
   reports `cpu`, the 11 checkpointed rows included. No `device_fallback` row exists.
2. **Single simulations are latency-bound on the GPU.** One hard-spike forward takes
   2.98–3.29 s on the GPU for 1 to 5000 motor neurons (14 000 steps at dt 0.1 ms),
   and one surrogate value-and-grad 6.45–7.24 s for 1 to 2000. The time does not
   grow with pool size: about 0.23 ms per step goes to per-step overhead, not
   arithmetic. The CPU is faster for one forward up to the largest pool it ran
   (1000 MN: 1.20 s vs 3.25 s) and for one gradient up to 100 MN; from 250 MN the
   GPU gradient is faster (1000 MN: 60.9 s vs 6.8 s, 8.9x).
3. **Batching pays on the GPU only for the forward.** GPU forward parallel
   efficiency stays 0.96–1.01 up to K = 1000 (310 sims/s vs CPU 181 sims/s, 1.7x).
   The GPU gradient batch runs out of memory at K = 256 (10 MN each; 19.1 GiB
   requested); up to K = 64 the CPU is faster (18.6 vs 9.2 sims/s).
   Opt-in gradient checkpointing removes every gradient OOM (K = 256 and 1000,
   5000 MN) at about 1.5x the time per call; the GPU gradient then beats the CPU
   from K = 1000 (84 vs 27 sims/s, 3.1x). See "Gradient checkpointing".
4. **Float32 surrogate signs are not reproducible across backends.** G2 at the
   same commit: CPU float32 0 agree / 19 wrong / 1 unresolved (the brief's
   "wrong in 19/20" reproduces, and matches the October CPU file exactly);
   GPU float32 19 / 0 / 1. Neither float32 run tracks the float64 gradients
   (CPU float32 has the opposite sign to x64 in 9 of 10 variants, GPU float32
   is 0.01–40x the x64 magnitude). Float32 is not qualified on either device.
   See "Float32 vs float64".
5. **Float64 (mixed) G2 is device-stable:** 17 / 2 / 1 on CPU and GPU, surrogate
   gradients within 4.6e-5 relative, CRN references identical. Matches the
   canonical October x64 file (17 / 2 / 1).
6. **Horizon (G1, `b_config`, x64 flag) reproduces on both devices:** dt 0.025 ms
   passes at 2 s (60 spikes, relative error 2.0e-6 CPU / 5.8e-6 GPU) and is
   FD-unconverged at 4 s (116 spikes); dt 0.1 ms is FD-unconverged at 2 s and 4 s.
   Identical verdicts and spike counts to the October file.
7. **Accepted fractions (G4 Stage A):** the figure uses the canonical October
   full run (surrogate joint 0.833, gCaL 0.958, w 0.667; CRN control 0.833 /
   0.917 / 0.667; gate 0.80; 0 non-finite; joint gate passes, w-only fails for
   both estimators). The full rerun at `4ed55623` (CPU, clean tree, `checks/cpu_full/`) reproduces it exactly: every accepted count per start and estimator, and all 24 cosines to four decimals.
8. **Known failure kept in view:** the G4 smoke (half duration, one case, 2 iters)
   gives surrogate joint 0.33 with cosines -0.998 (S1) and -0.973 (S2) on both
   CPU and GPU (identical to 1e-10); CRN control 0.83. The October smoke (older
   schema) also rejected the surrogate steps at S1, so it is qualitatively
   consistent. The canonical full run has one near-opposite cosine (-0.9997,
   case conn 0 / offset 10, S2).

## Revision, machine, environment

| Item | Value |
|---|---|
| Benchmark commit | `4ed55623` on `bench/jaxley-gpu-scaling` (= PR #23 head `7869facc` + runner, plot and tests); tree clean during every run |
| Post-run code changes | `derive()` baseline lookup fixed (speed-up / efficiency were empty for K > 1; post-processing only, raw rows unchanged); figure layout. Commit that adds this note |
| Host | Native Ubuntu 26.04.1 LTS workstation, kernel 6.14.0, **not WSL2** (deviation from the brief's laptop setup) |
| CPU / RAM | AMD Ryzen 9 7950X, 16 cores / 32 threads, 91 GiB |
| GPU | NVIDIA GeForce RTX 4090, 24 564 MiB, driver 595.71.05, compute capability 8.9, driver CUDA 13.2 |
| JAX devices | `jax.devices()` = `[CudaDevice(id=0)]`; with `JAX_PLATFORMS=cpu`: `[CpuDevice(id=0)]` |
| Software | Python 3.12.8, jax 0.9.0, jaxlib 0.9.0, jax-cuda12-plugin/pjrt 0.9.0, jaxley 0.13.0, numpy 2.4.6, psutil 7.2.2, neuron 8.2.7 (tests only), uv 0.12.10 |
| Allocator | `XLA_PYTHON_CLIENT_PREALLOCATE=false` for every child |

Environment setup (isolated worktree and venv; `uv.lock` untouched, nothing
CUDA-related committed):

```bash
cd ~/code/MyoGen && git fetch origin feat/jaxley
git worktree add --detach ~/code/MyoGen-jaxley-fig2 7869facc
cd ~/code/MyoGen-jaxley-fig2
uv sync
uv pip install "jax[cuda12]==0.9.0"
uv pip install "neuron==8.2.7"          # test_twin_harness NEURON smoke test
PATH=$PWD/.venv/bin:$PATH python -c "from myogen import _setup_myogen; _setup_myogen()"
# benchmark branch delivered as a git bundle, then
git fetch /tmp/bench.bundle bench/jaxley-gpu-scaling:refs/remotes/bundle/bench
git reset --hard bundle/bench           # 4ed55623
```

## Commands

Run from `experiments/twin/`, in order, nothing else on the machine during the
benchmarks (`gpu_benchmark_run_20261117/run_pipeline.sh`):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python gpu_benchmark.py --timeout-s 3600 --out-dir results/gpu_benchmark_run/bench
python gpu_benchmark.py --sweeps stress --devices gpu --timeout-s 3600 \
    --out-dir results/gpu_benchmark_run/stress
# then for JAX_PLATFORMS=cpu and for the GPU (JAX_PLATFORMS unset):
python gradient_repair.py --tests g2
python gradient_repair.py --tests g2 --x64
python gradient_repair.py --tests g1 --x64 --regime b_config
python g4_descent.py --smoke --iters 2
```

Checkpointed gradient rows (`remat/run_remat_bench.sh`, revision `ad9afd97`,
alone on the machine):

```bash
python gpu_benchmark.py --devices gpu --sweeps batch pool stress \
    --pool-sizes 10 100 1000 --ops surrogate_value_and_grad --remat-segment 140 \
    --timeout-s 3600 --out-dir results/gpu_benchmark_remat
# equality checks, once with JAX_PLATFORMS=cpu and once on the GPU:
PYTHONPATH=. python evidence/gpu_benchmark_run_20261117/remat/remat_g1.py
PYTHONPATH=. python evidence/gpu_benchmark_run_20261117/remat/remat_equal.py
# the pool sizes the run above skipped (remat/pool2/run_remat_pool2.sh, 567bb031):
python gpu_benchmark.py --devices gpu --sweeps pool --pool-sizes 1 50 250 500 \
    --ops surrogate_value_and_grad --remat-segment 140 --timeout-s 3600 \
    --out-dir results/gpu_benchmark_remat_pool2
```

Follow-ups: `run_g2_repeat.sh` (float32 G2 twice more on the GPU, once more on
the CPU) and `run_g4_full.sh` (`g4_descent.py --iters 10`, CPU).
Preflight (before the run): `--theta default|perturbed` timing check at
n_mn = 10, and `--audit float32|float64` (graph dtype audit of the G1/G2
objectives).

Protocol: step drive, dt 0.1 ms, t_pre 0.2 s, t_on 1.0 s, t_post 0.2 s, 40 Hz,
n_dd 100, seed 0 (14 000 steps, 1.4 s). Pool sweep n_mn 1–1000 at K = 1; batch
sweep K 1–1000 at n_mn 10; stress n_mn 2000, 5000 (GPU); precision: G4 case
(conn 0, offset 0, truth (1.5, 0.011), start (1.0, 0.009)), n_mn 8, dt 0.025 ms.
Each configuration is a fresh subprocess; build, first call (compile + first run)
and 2 warm-ups are untimed for the statistics; 7 repetitions, each synchronized
with `jax.block_until_ready` before `perf_counter` stops.

## Files

| Path (under `experiments/twin/evidence/`) | Content |
|---|---|
| `gpu_benchmark_20261117_144350.json` | Main benchmark: manifest + 60 rows (the run's `bench/benchmark.json`; workstation sha256 `25a825dd1622a2df…`; the committed copy has the home directory replaced by `~` in two traceback tails, nothing else) |
| `gpu_benchmark_20261117_164705.json` | GPU stress rows (the run's `stress/benchmark.json`; sha256 `be944c29e7f7edf4…`; same substitution) |
| `gpu_benchmark_20261117_195639.json` | Checkpointed GPU gradient rows (`remat_segment` 140), revision `ad9afd97`, 11 rows, all ok (sha256 `8ee0b92dfde86f84…`, no substitution needed) |
| `gpu_benchmark_20261117_213319.json` | Checkpointed GPU gradient at 1, 50, 250, 500 MN, revision `567bb031`, 4 rows, all ok (sha256 `dd552111e35fc25b…`) |
| `gpu_benchmark_run_20261117/remat/` | Its tidy CSV, log, script, `nvidia-smi`; `remat_g1.py`, `remat_equal.py` and their outputs in `checks/` |
| `gpu_benchmark_run_20261117/bench/benchmark_reps.csv`, `stress/benchmark_reps.csv` | Tidy CSV, one row per repetition; failed rows kept |
| `gpu_benchmark_run_20261117/checks/{cpu,gpu}/` | G2 f32, G2 x64, G1 b_config, G4 smoke evidence + logs per device |
| `gpu_benchmark_run_20261117/checks/repeat/` | Float32 G2 repeats |
| `gpu_benchmark_run_20261117/checks/cpu_full/` | Full G4 rerun (CPU, same revision) |
| `gpu_benchmark_run_20261117/preflight/` | Theta check, graph dtype audits |
| `gpu_benchmark_run_20261117/*.sh, *.log, nvidia_smi_*.txt, commit.txt` | Scripts, logs, GPU state before/after |
| `results/figures/figure_computational_gradient_scaling.{png,pdf}`, `_source_data.json` (repo root) | Draft figure and every plotted value with input hashes |

Regenerate the figure (reads only these files; uses `simpetru_figstyle` if
installed, plain Matplotlib otherwise):

```bash
E=experiments/twin/evidence; R=$E/gpu_benchmark_run_20261117
python scripts/plot_computational_gradient_scaling.py \
  --benchmark $E/gpu_benchmark_20261117_144350.json $E/gpu_benchmark_20261117_164705.json \
              $E/gpu_benchmark_20261117_195639.json $E/gpu_benchmark_20261117_213319.json \
  --g2 "Float32, CPU=$R/checks/cpu/g2_f32__gradient_repair_f32_20261117_182119.json" \
       "Float32, GPU=$R/checks/gpu/g2_f32__gradient_repair_f32_20261117_182735.json" \
       "Mixed x64, CPU=$R/checks/cpu/g2_x64__gradient_repair_x64_20261117_182152.json" \
       "Mixed x64, GPU=$R/checks/gpu/g2_x64__gradient_repair_x64_20261117_183200.json" \
  --g1 $R/checks/cpu/g1_x64_bconfig__gradient_repair_x64_20261117_182306.json \
  --g4 $E/g4_descent_full_20261018_180008.json
```

Panels: A time per call, four small plots (hard-spike forward and surrogate
value-and-grad, one pool over pool size and batched pools over batch size;
failed runs as crosses on the top edge; the GPU gradient line is the
checkpointed run, the plain rows with their OOMs are in the tables); B G2 sign,
one row per precision and device, a dot pair (gCaL, w) per variant; C G1 horizon; D G4 accepted fractions, a dot
per start and a bar for all steps; E G4 direction cosines.

## Tests

`tests/test_gpu_benchmark.py` (16 tests): synchronization before the clock stops;
compile and steady state stored apart; >= 2 warm-ups / >= 7 repetitions enforced;
CPU and GPU rows cannot overwrite each other; precision recorded and mixed graphs
labelled mixed; K = 1 default / K > 1 draws; device fallback counts as failure;
OOM classification; timeout row; failed / OOM / skipped rows kept in JSON and CSV;
manifest schema; the plot reads only saved evidence (no jax, jaxley, poolbuild
or runner imported); batch speed-up uses the K = 1 row (default theta);
checkpointed rows are kept apart from plain rows; traceback tails carry no home
directory.

`tests/test_checkpointed_scan.py` (2 tests): a toy scan gives the same value and
gradient for segments 1, 12, 40 and 120 and rejects a segment that does not divide
the steps; a 2-MN `PoolTwin` (3000 steps) gives the same mean-voltage value and
gradient with and without checkpointing (rtol 1e-4).

Result at `f661376e` on the workstation (GPU visible):
`pytest tests/test_checkpointed_scan.py tests/test_gpu_benchmark.py tests/test_twin_harness.py`
gives 57 passed in 95 s. On the laptop (no jax) `test_manifest_schema` cannot import
the runner and fails; the other benchmark tests pass.

## Benchmark table

Throughput definitions: sims/s = K / median; bio-s/wall-s = K x 1.4 s / median;
MN·steps/s = K x n_mn x 14 000 / median. Build = Python-side model construction
(grows linearly with pool size, 1944 s at 5000 MN; outside both timing columns).
Peak device memory is the XLA allocator peak for the process (GPU only); CPU rows
have observed RSS only, not a peak. Grad/fwd = gradient median / forward median at
the same shape. Speed-up and efficiency are recomputed from the saved rows with
the fixed `derive()`.

| Sweep | Device | Op | Precision | N MN | K | Status | Build (s) | First call (s) | Median (s) | IQR (s) | Min–max (s) | Sims/s | Bio-s/wall-s | MN·steps/s | Grad/fwd | Speedup vs K=1 | Parallel eff. | Peak device mem (GB) | RSS after (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pool | cpu | hard_forward | float32 | 1 | 1 | ok | 4.1 | 1.20 | 0.475 | 0.011 | 0.456–0.485 | 2.1 | 2.94 | 2.94e+04 |  |  |  |  | 1.23 |
| pool | cpu | hard_forward | float32 | 10 | 1 | ok | 7.4 | 1.20 | 0.497 | 0.013 | 0.491–0.521 | 2.01 | 2.81 | 2.81e+05 |  |  |  |  | 1.40 |
| pool | cpu | hard_forward | float32 | 50 | 1 | ok | 17.4 | 1.23 | 0.538 | 0.005 | 0.534–0.559 | 1.86 | 2.6 | 1.3e+06 |  |  |  |  | 1.55 |
| pool | cpu | hard_forward | float32 | 100 | 1 | ok | 30.2 | 1.26 | 0.569 | 0.010 | 0.555–0.577 | 1.76 | 2.46 | 2.46e+06 |  |  |  |  | 1.72 |
| pool | cpu | hard_forward | float32 | 250 | 1 | ok | 68.4 | 1.69 | 0.670 | 0.003 | 0.644–0.704 | 1.49 | 2.09 | 5.22e+06 |  |  |  |  | 2.24 |
| pool | cpu | hard_forward | float32 | 500 | 1 | ok | 133.5 | 1.55 | 0.835 | 0.016 | 0.807–0.845 | 1.2 | 1.68 | 8.39e+06 |  |  |  |  | 3.07 |
| pool | cpu | hard_forward | float32 | 1000 | 1 | ok | 261.5 | 1.92 | 1.197 | 0.007 | 1.186–1.226 | 0.836 | 1.17 | 1.17e+07 |  |  |  |  | 4.43 |
| pool | cpu | surrogate_value_and_grad | float32 | 1 | 1 | ok | 4.1 | 1.91 | 0.679 | 0.021 | 0.670–0.712 | 1.47 | 2.06 | 2.06e+04 | 1.43 |  |  |  | 1.29 |
| pool | cpu | surrogate_value_and_grad | float32 | 10 | 1 | ok | 7.4 | 2.02 | 0.802 | 0.033 | 0.786–0.845 | 1.25 | 1.74 | 1.74e+05 | 1.61 |  |  |  | 1.52 |
| pool | cpu | surrogate_value_and_grad | float32 | 50 | 1 | ok | 17.4 | 3.34 | 2.154 | 0.034 | 2.114–2.215 | 0.464 | 0.65 | 3.25e+05 | 4.00 |  |  |  | 1.96 |
| pool | cpu | surrogate_value_and_grad | float32 | 100 | 1 | ok | 30.0 | 4.78 | 3.802 | 0.347 | 3.471–3.917 | 0.263 | 0.368 | 3.68e+05 | 6.68 |  |  |  | 2.52 |
| pool | cpu | surrogate_value_and_grad | float32 | 250 | 1 | ok | 68.3 | 9.45 | 8.227 | 0.088 | 8.167–8.306 | 0.122 | 0.17 | 4.25e+05 | 12.28 |  |  |  | 4.20 |
| pool | cpu | surrogate_value_and_grad | float32 | 500 | 1 | ok | 133.4 | 21.22 | 19.360 | 0.555 | 19.152–20.449 | 0.0517 | 0.0723 | 3.62e+05 | 23.19 |  |  |  | 6.99 |
| pool | cpu | surrogate_value_and_grad | float32 | 1000 | 1 | ok | 262.4 | 62.29 | 60.879 | 0.464 | 60.443–61.305 | 0.0164 | 0.023 | 2.3e+05 | 50.87 |  |  |  | 12.28 |
| batch | cpu | hard_forward | float32 | 10 | 1 | ok | 7.4 | 1.21 | 0.537 | 0.007 | 0.529–0.543 | 1.86 | 2.61 | 2.61e+05 |  | 1.0 | 1.00 |  | 1.40 |
| batch | cpu | hard_forward | float32 | 10 | 4 | ok | 7.3 | 1.26 | 0.549 | 0.006 | 0.542–0.564 | 7.29 | 10.2 | 1.02e+06 |  | 3.9 | 0.98 |  | 1.40 |
| batch | cpu | hard_forward | float32 | 10 | 16 | ok | 7.4 | 1.31 | 0.582 | 0.026 | 0.575–0.612 | 27.5 | 38.5 | 3.85e+06 |  | 14.7 | 0.92 |  | 1.44 |
| batch | cpu | hard_forward | float32 | 10 | 64 | ok | 7.3 | 1.60 | 0.901 | 0.030 | 0.885–0.932 | 71 | 99.5 | 9.95e+06 |  | 38.1 | 0.60 |  | 1.44 |
| batch | cpu | hard_forward | float32 | 10 | 256 | ok | 7.4 | 2.70 | 1.902 | 0.024 | 1.888–1.970 | 135 | 188 | 1.88e+07 |  | 72.2 | 0.28 |  | 1.68 |
| batch | cpu | hard_forward | float32 | 10 | 1000 | ok | 7.4 | 6.19 | 5.511 | 0.057 | 5.477–5.622 | 181 | 254 | 2.54e+07 |  | 97.4 | 0.10 |  | 2.51 |
| batch | cpu | surrogate_value_and_grad | float32 | 10 | 1 | ok | 7.4 | 2.01 | 0.812 | 0.009 | 0.805–0.828 | 1.23 | 1.72 | 1.72e+05 | 1.51 | 1.0 | 1.00 |  | 1.51 |
| batch | cpu | surrogate_value_and_grad | float32 | 10 | 4 | ok | 7.4 | 2.33 | 1.079 | 0.043 | 1.052–1.120 | 3.71 | 5.19 | 5.19e+05 | 1.97 | 3.0 | 0.75 |  | 1.78 |
| batch | cpu | surrogate_value_and_grad | float32 | 10 | 16 | ok | 7.5 | 2.91 | 1.753 | 0.031 | 1.711–1.820 | 9.13 | 12.8 | 1.28e+06 | 3.01 | 7.4 | 0.46 |  | 2.85 |
| batch | cpu | surrogate_value_and_grad | float32 | 10 | 64 | ok | 7.4 | 4.64 | 3.445 | 0.039 | 3.417–3.506 | 18.6 | 26 | 2.6e+06 | 3.82 | 15.1 | 0.24 |  | 6.93 |
| batch | cpu | surrogate_value_and_grad | float32 | 10 | 256 | ok | 7.4 | 11.26 | 10.821 | 0.131 | 10.676–10.890 | 23.7 | 33.1 | 3.31e+06 | 5.69 | 19.2 | 0.08 |  | 23.29 |
| batch | cpu | surrogate_value_and_grad | float32 | 10 | 1000 | ok | 7.4 | 57.55 | 37.504 | 0.186 | 37.344–37.662 | 26.7 | 37.3 | 3.73e+06 | 6.80 | 21.7 | 0.02 |  | 84.24 |
| precision | cpu | hard_forward_loss | float32 | 8 | 1 | ok | 11.7 | 1.13 | 0.457 | 0.011 | 0.446–0.469 | 2.19 | 8.76 | 2.8e+06 |  |  |  |  | 1.77 |
| precision | cpu | surrogate_grad | float32 | 8 | 1 | ok | 11.5 | 4.34 | 2.859 | 0.028 | 2.835–2.914 | 0.35 | 1.4 | 4.48e+05 | 6.26 |  |  |  | 2.53 |
| precision | cpu | hard_forward_loss | mixed | 8 | 1 | ok | 11.7 | 1.72 | 0.550 | 0.014 | 0.539–0.585 | 1.82 | 7.28 | 2.33e+06 |  |  |  |  | 2.06 |
| precision | cpu | surrogate_grad | mixed | 8 | 1 | ok | 11.7 | 5.86 | 4.039 | 0.050 | 3.998–4.121 | 0.248 | 0.99 | 3.17e+05 | 7.35 |  |  |  | 3.51 |
| pool | gpu | hard_forward | float32 | 1 | 1 | ok | 5.7 | 5.65 | 2.984 | 0.020 | 2.961–3.019 | 0.335 | 0.469 | 4.69e+03 |  |  |  | 0.10 | 1.49 |
| pool | gpu | hard_forward | float32 | 10 | 1 | ok | 10.4 | 5.82 | 3.198 | 0.023 | 3.190–3.224 | 0.313 | 0.438 | 4.38e+04 |  |  |  | 0.10 | 1.66 |
| pool | gpu | hard_forward | float32 | 50 | 1 | ok | 24.9 | 5.85 | 3.186 | 0.016 | 3.161–3.193 | 0.314 | 0.439 | 2.2e+05 |  |  |  | 0.10 | 1.81 |
| pool | gpu | hard_forward | float32 | 100 | 1 | ok | 43.0 | 5.75 | 3.087 | 0.025 | 3.082–3.114 | 0.324 | 0.454 | 4.54e+05 |  |  |  | 0.10 | 2.00 |
| pool | gpu | hard_forward | float32 | 250 | 1 | ok | 95.7 | 5.83 | 3.159 | 0.005 | 3.148–3.165 | 0.317 | 0.443 | 1.11e+06 |  |  |  | 0.10 | 2.54 |
| pool | gpu | hard_forward | float32 | 500 | 1 | ok | 190.0 | 5.99 | 3.236 | 0.004 | 3.231–3.250 | 0.309 | 0.433 | 2.16e+06 |  |  |  | 0.10 | 3.44 |
| pool | gpu | hard_forward | float32 | 1000 | 1 | ok | 373.4 | 5.90 | 3.253 | 0.008 | 3.223–3.261 | 0.307 | 0.43 | 4.3e+06 |  |  |  | 0.15 | 4.95 |
| pool | gpu | surrogate_value_and_grad | float32 | 1 | 1 | ok | 5.6 | 13.97 | 7.240 | 0.011 | 7.234–7.249 | 0.138 | 0.193 | 1.93e+03 | 2.43 |  |  | 0.10 | 1.65 |
| pool | gpu | surrogate_value_and_grad | float32 | 10 | 1 | ok | 10.5 | 11.56 | 6.455 | 0.006 | 6.447–6.461 | 0.155 | 0.217 | 2.17e+04 | 2.02 |  |  | 0.13 | 1.74 |
| pool | gpu | surrogate_value_and_grad | float32 | 50 | 1 | ok | 25.0 | 11.61 | 6.452 | 0.004 | 6.446–6.462 | 0.155 | 0.217 | 1.08e+05 | 2.03 |  |  | 0.38 | 1.91 |
| pool | gpu | surrogate_value_and_grad | float32 | 100 | 1 | ok | 42.3 | 11.90 | 6.700 | 0.008 | 6.695–6.708 | 0.149 | 0.209 | 2.09e+05 | 2.17 |  |  | 0.74 | 2.11 |
| pool | gpu | surrogate_value_and_grad | float32 | 250 | 1 | ok | 95.7 | 11.84 | 6.548 | 0.003 | 6.542–6.550 | 0.153 | 0.214 | 5.35e+05 | 2.07 |  |  | 1.85 | 2.68 |
| pool | gpu | surrogate_value_and_grad | float32 | 500 | 1 | ok | 187.5 | 11.85 | 6.543 | 0.006 | 6.537–6.555 | 0.153 | 0.214 | 1.07e+06 | 2.02 |  |  | 3.69 | 3.65 |
| pool | gpu | surrogate_value_and_grad | float32 | 1000 | 1 | ok | 370.1 | 12.01 | 6.806 | 0.004 | 6.794–6.810 | 0.147 | 0.206 | 2.06e+06 | 2.09 |  |  | 7.38 | 5.30 |
| batch | gpu | hard_forward | float32 | 10 | 1 | ok | 10.4 | 5.80 | 3.145 | 0.015 | 3.138–3.168 | 0.318 | 0.445 | 4.45e+04 |  | 1.0 | 1.00 | 0.10 | 1.67 |
| batch | gpu | hard_forward | float32 | 10 | 4 | ok | 10.6 | 6.23 | 3.232 | 0.013 | 3.222–3.247 | 1.24 | 1.73 | 1.73e+05 |  | 3.9 | 0.97 | 0.10 | 1.77 |
| batch | gpu | hard_forward | float32 | 10 | 16 | ok | 10.3 | 6.19 | 3.167 | 0.011 | 3.158–3.174 | 5.05 | 7.07 | 7.07e+05 |  | 15.9 | 0.99 | 0.10 | 1.76 |
| batch | gpu | hard_forward | float32 | 10 | 64 | ok | 10.4 | 6.19 | 3.108 | 0.001 | 3.102–3.111 | 20.6 | 28.8 | 2.88e+06 |  | 64.8 | 1.01 | 0.14 | 1.78 |
| batch | gpu | hard_forward | float32 | 10 | 256 | ok | 10.5 | 6.29 | 3.276 | 0.005 | 3.271–3.281 | 78.1 | 109 | 1.09e+07 |  | 245.8 | 0.96 | 0.68 | 1.77 |
| batch | gpu | hard_forward | float32 | 10 | 1000 | ok | 10.3 | 6.53 | 3.228 | 0.004 | 3.223–3.233 | 310 | 434 | 4.34e+07 |  | 974.4 | 0.97 | 1.70 | 1.89 |
| batch | gpu | surrogate_value_and_grad | float32 | 10 | 1 | ok | 10.4 | 11.53 | 6.518 | 0.007 | 6.514–6.523 | 0.153 | 0.215 | 2.15e+04 | 2.07 | 1.0 | 1.00 | 0.13 | 1.75 |
| batch | gpu | surrogate_value_and_grad | float32 | 10 | 4 | ok | 10.6 | 12.45 | 6.631 | 0.007 | 6.627–6.655 | 0.603 | 0.845 | 8.45e+04 | 2.05 | 3.9 | 0.98 | 0.32 | 1.86 |
| batch | gpu | surrogate_value_and_grad | float32 | 10 | 16 | ok | 10.4 | 12.95 | 6.825 | 0.003 | 6.824–6.828 | 2.34 | 3.28 | 3.28e+05 | 2.16 | 15.3 | 0.95 | 1.28 | 1.92 |
| batch | gpu | surrogate_value_and_grad | float32 | 10 | 64 | ok | 10.4 | 12.90 | 6.974 | 0.034 | 6.941–6.982 | 9.18 | 12.8 | 1.28e+06 | 2.24 | 59.8 | 0.93 | 5.13 | 2.15 |
| batch | gpu | surrogate_value_and_grad | float32 | 10 | 256 | oom |  |  |  |  |  |  |  |  |  |  |  |  |  |
| batch | gpu | surrogate_value_and_grad | float32 | 10 | 1000 | not_attempted |  |  |  |  |  |  |  |  |  |  |  |  |  |
| precision | gpu | hard_forward_loss | float32 | 8 | 1 | ok | 40.5 | 10.47 | 9.298 | 0.012 | 9.281–9.303 | 0.108 | 0.43 | 1.38e+05 |  |  |  | 0.34 | 2.69 |
| precision | gpu | surrogate_grad | float32 | 8 | 1 | ok | 40.2 | 53.42 | 46.282 | 0.007 | 46.272–46.287 | 0.0216 | 0.0864 | 2.77e+04 | 4.98 |  |  | 0.85 | 2.92 |
| precision | gpu | hard_forward_loss | mixed | 8 | 1 | ok | 47.3 | 13.69 | 11.265 | 0.008 | 11.246–11.266 | 0.0888 | 0.355 | 1.14e+05 |  |  |  | 0.71 | 2.86 |
| precision | gpu | surrogate_grad | mixed | 8 | 1 | ok | 47.1 | 67.55 | 54.798 | 0.002 | 54.789–54.801 | 0.0182 | 0.073 | 2.34e+04 | 4.86 |  |  | 1.76 | 3.30 |
| stress | gpu | hard_forward | float32 | 2000 | 1 | ok | 754.4 | 6.03 | 3.291 | 0.001 | 3.286–3.295 | 0.304 | 0.425 | 8.51e+06 |  |  |  | 0.29 | 6.07 |
| stress | gpu | hard_forward | float32 | 5000 | 1 | ok | 1944.1 | 5.99 | 3.267 | 0.007 | 3.261–3.275 | 0.306 | 0.428 | 2.14e+07 |  |  |  | 0.68 | 8.98 |
| stress | gpu | surrogate_value_and_grad | float32 | 2000 | 1 | ok | 742.4 | 12.07 | 6.759 | 0.008 | 6.745–6.762 | 0.148 | 0.207 | 4.14e+06 | 2.05 |  |  | 14.75 | 6.68 |
| stress | gpu | surrogate_value_and_grad | float32 | 5000 | 1 | oom |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Gradient values of the timing workload

The benchmark's gradient workload is `phase2_benchmark.py`'s loss (mean squared
smoothed surrogate rate over 1.4 s at dt 0.1 ms, float32). It times reverse mode
at scale; its values are not a qualified gradient. At the default parameters the
gradient magnitudes are 4e9 to 8e20, and every perturbed batch with K >= 4 has
non-finite entries (`output_summary.all_finite = false`), on CPU and GPU alike.
The timings are unaffected (same arithmetic either way). The qualified gradient
evidence is G1, G2 (x64) and G4 below, not this workload.

## Gradient checkpointing

Why the GPU gradient ran out of memory: reverse mode stores the state of every
one of the 14 000 time steps for the backward pass, so memory grows with steps x
neurons x batch (7.4 GB at 1000 MN, 14.8 GB at 2000 MN, 5.1 GB at K 64).
`TwinSpec.remat_segment` (default `None`, i.e. off; added in `ad9afd97`) splits
the time loop into segments under `jax.checkpoint`: only the state at segment
boundaries is kept and each segment is recomputed during the backward pass. The
equations, parameters, losses and estimators are unchanged; it trades time for
memory. All results elsewhere in this note use the default (off).

Benchmark (GPU, float32, `remat_segment` 140 = 100 segments, same protocol and
theta rule as the main table; peak = XLA allocator peak, compilation included):

| Configuration | Plain GPU: median, peak | Checkpointed GPU: median, peak | CPU median |
|---|---|---|---|
| 1 pool, 1 MN | 7.24 s, 0.10 GB | 9.78 s, 0.10 GB | 0.68 s |
| 1 pool, 10 MN | 6.46 s, 0.14 GB | 9.68 s, 0.10 GB | 0.80 s |
| 1 pool, 50 MN | 6.45 s, 0.38 GB | 9.96 s, 0.10 GB | 2.15 s |
| 1 pool, 100 MN | 6.70 s, 0.75 GB | 9.75 s, 0.10 GB | 3.80 s |
| 1 pool, 250 MN | 6.55 s, 1.85 GB | 9.78 s, 0.10 GB | 8.23 s |
| 1 pool, 500 MN | 6.54 s, 3.69 GB | 9.72 s, 0.15 GB | 19.36 s |
| 1 pool, 1000 MN | 6.81 s, 7.38 GB | 10.11 s, 0.29 GB | 60.88 s |
| 1 pool, 2000 MN | 6.76 s, 14.76 GB | 9.95 s, 0.40 GB | not run |
| 1 pool, 5000 MN | **OOM** (34.23 GiB requested) | 10.47 s, 1.01 GB | not run |
| K 1 x 10 MN | 6.52 s, 0.14 GB | 9.89 s, 0.10 GB | 0.81 s |
| K 4 | 6.63 s, 0.32 GB | 9.95 s, 0.14 GB | 1.08 s |
| K 16 | 6.83 s, 1.28 GB | 10.12 s, 0.14 GB | 1.75 s |
| K 64 | 6.97 s, 5.13 GB | 10.01 s, 0.14 GB | 3.45 s |
| K 256 | **OOM** (19.10 GiB requested) | 10.52 s, 0.54 GB | 10.82 s |
| K 1000 | not attempted | 11.93 s, 1.86 GB | 37.50 s |

Time per call rises 1.35-1.54x (each segment runs forward twice); peak memory
falls up to 38x. The GPU gradient stays latency-bound (about 10 s for every
size). Against the CPU it ties at K 256 and wins at K 1000 (3.1x). In one pool it
is slower than the CPU up to 250 MN and faster from 500 MN (2.0x; 1000 MN 6.0x,
against 8.9x without checkpointing). The 1, 50, 250 and 500 MN rows come from a
second solo run at `567bb031` (same runner and protocol). As in the plain
rows, every perturbed batch with K >= 4 has non-finite gradient entries: these
rows time the workload, they are not usable gradients (see "Gradient values of
the timing workload").

An earlier K 64 check (`gpu_benchmark_20261117_194833.json` on the workstation,
not copied) ran while the G4 rerun was using the machine; the solo run above
supersedes it (10.0 s, 0.135 GB, the same peak memory).

Same gradient where the gradient is qualified: in the G1 `b_config` cell that
passes (dt 0.025 ms, 2 s, x64, Salome's G1 loss unchanged, `remat_segment` 400),
plain and checkpointed agree to 3e-14 (CPU) / 8e-15 (GPU) in the loss and
1.7e-9 / 4.2e-8 relative in the gradient, with 60 spikes in both
(`remat/checks/remat_g1_{cpu,gpu}.json`). Unit tests: `tests/test_checkpointed_scan.py`.

Not a test at the benchmark shape: at float32, 14 000 steps, plain and
checkpointed compile to different programs, and the spiking dynamics amplify the
different rounding. On the GPU the loss is identical and the gradients differ by
0.7-14 %; on the CPU the loss already differs in the third digit (4.925 vs 4.939)
and gradients differ up to 100x, including sign. Plain CPU and plain GPU differ
by the same order, with opposite signs for the mean-voltage gradient
(`remat/checks/remat_equal_{cpu,gpu}.json`). This is consistent with the float32
problem reported below rather than an error of checkpointing, given the x64
agreement above; no x64 run at this shape was made. (`remat_equal.py` is the
script as run; its `mode` variable is unused and both losses go through the
surrogate path.)

Use it for gradients that do not fit; the plain scan stays the default.

## Float32 vs float64

Graph dtype audit (`jax.make_jaxpr`, nested jaxprs included): under float32 every
float in the G1/G2 objectives and the benchmark workloads is float32. Under
`--x64` the G1 `b_config` objective still has 22 float32 values in its graph and
G2 27–35, and the G4 precision rows keep `theta['drive_gain']`,
`theta['gNaP_scale']` and `theta['w_inh']` in float32 (`PoolTwin` casts in
`_run` / `simulate_events`). These runs are labelled **mixed**, never float64.

Cost of mixed x64 (G4 case, n_mn 8, dt 0.025 ms, K = 1): CPU forward 0.457 →
0.550 s (1.20x), gradient 2.86 → 4.04 s (1.41x); GPU forward 9.30 → 11.27 s
(1.21x), gradient 46.3 → 54.8 s (1.18x). The GPU is 14–20x slower than the CPU
on this small case.

G2 (surrogate sign vs CRN finite-scale control direction, 10 variants x 2
coordinates):

| Run | Agree | Wrong | Unresolved | Surrogate magnitude range |
|---|---|---|---|---|
| October CPU float32 (`322cf9b5`, trimmed in `5a4fa8db`) | 0 | 19 | 1 | |
| CPU float32 | 0 | 19 | 1 | 4.5e9 – 1.9e11 |
| GPU float32 | 19 | 0 | 1 | 1.9e8 – 1.2e9 |
| October CPU x64 (canonical) | 17 | 2 | 1 | |
| CPU x64 (mixed) | 17 | 2 | 1 | 9.0e6 – 1.5e10 |
| GPU x64 (mixed) | 17 | 2 | 1 | 9.0e6 – 1.5e10 |

Float32 repeats at the same commit (`checks/repeat/`): two more GPU runs and one
more CPU run give bit-identical surrogate gradients and the same counts (GPU
19 / 0 / 1 three times, CPU 0 / 19 / 1 twice). The float32 result is
deterministic per device, so the difference is between backends, not run-to-run
noise.

Reading: the float32 surrogate gradient is dominated by rounding. Ratio
float32 / x64 per variant (same device): CPU -1.5 to -3.5 in 8 variants, +0.9 at
`vth=51` and -2.2e4 at `dt=0.05`; GPU +0.1 to +0.7 in 8 variants, -0.01 at
`vth=51` and +40 at `dt=0.05`. At `vth=51` x64 says wrong and GPU float32 says
agree.
The float32 CRN reference itself moves (w: -4321 CPU, -3909 GPU; x64 -4321 on
both). The GPU float32 "agree" is therefore not evidence of a valid gradient.
The x64 (mixed) results agree across devices to 4.6e-5.

## Horizon (G1, `b_config`)

| dt (ms) | Horizon (s) | Spikes | CPU | GPU | October |
|---|---|---|---|---|---|
| 0.025 | 2 | 60 | PASS (rel 2.0e-6) | PASS (rel 5.8e-6) | PASS |
| 0.025 | 4 | 116 | FD_UNCONVERGED | FD_UNCONVERGED | FD_UNCONVERGED |
| 0.1 | 2 | 60 | FD_UNCONVERGED | FD_UNCONVERGED | FD_UNCONVERGED |
| 0.1 | 4 | 116 | FD_UNCONVERGED | FD_UNCONVERGED | FD_UNCONVERGED |

FD_UNCONVERGED means the local finite-difference plateau was not found; it is
reported as unresolved, not as a failure of AD. No cell is FAILED.

## Accepted fractions (G4 Stage A, hard-spike loss judges)

8 cases (2 connectivity seeds x 2 event offsets x 2 true parameter sets) x 3
starts x 2 step sizes (lambda 0.03, 0.1). Identical in the October canonical file
and the November rerun.

| Estimator | Joint | gCaL only | w only | Non-finite | Joint gate (0.80) |
|---|---|---|---|---|---|
| Surrogate | 0.833 (40/48) | 0.958 (23/24) | 0.667 (16/24) | 0 | pass |
| CRN control | 0.833 (40/48) | 0.917 (22/24) | 0.667 (16/24) | 0 | pass |

Per start (both estimators): S1 (both wrong) and S3 (w wrong) accept every
joint and w step; S2 (gCaL wrong, w already true) accepts 8/16 joint steps and
0/8 w-only steps. All of the joint and w-only shortfall comes from S2, where a w
step cannot lower the loss. gCaL-only misses: surrogate 1, CRN control 2, both
at S2.

Cosine surrogate vs CRN control: median 0.999, 23 of 24 above 0.77, one
near-opposite at -0.9997 (conn 0, offset 10, truth (1.5, 0.011), S2).

## Out-of-memory, failures, skipped

| Row | Status | Detail |
|---|---|---|
| batch, GPU, surrogate value-and-grad, n_mn 10, K 256 | oom | `RESOURCE_EXHAUSTED`: 19.10 GiB requested |
| batch, GPU, surrogate value-and-grad, n_mn 10, K 1000 | not_attempted | follows the OOM at K 256 |
| stress, GPU, surrogate value-and-grad, n_mn 5000 | oom | `RESOURCE_EXHAUSTED`: 34.23 GiB requested |

All three rows run with gradient checkpointing (separate rows, above); the
plain rows stay as recorded. No timeouts (limit 3600 s per configuration), no
other failures, no device fallback. GPU gradient peak memory grows linearly with the work per call: 7.4 GB
at 1000 MN, 14.8 GB at 2000 MN, 5.1 GB at K 64 x 10 MN.

## Limitations

- No CPU rows above 1000 MN (the stress sweep is GPU-only), so panels A/B make no
  CPU-GPU comparison beyond 1000 MN.
- CPU memory is observed RSS, not a peak; GPU peak is process-wide and includes
  compilation.
- The canonical G4 file was produced at `a483b1d` (dirty) with numpy 2.3.5; its
  recorded source hashes differ from HEAD for `poolbuild.py`, `observables.py`,
  `gradient_repair.py` and `motor_neurons.py`. The November full rerun at the current commit reproduces it exactly, so the drift does not change G4.
- In the canonical G4 run, S2 starts with w at its true value, so S2 w-only steps
  (0/8 accepted) cannot reduce the loss; part of the w-only shortfall is that.
- One-spike CPU-GPU difference in float32 forward output at n_mn 10 (36 vs 37
  spikes); identical at 1, 100 and 1000 MN.
- K = 1 benchmark rows use `default_theta()`; K > 1 rows use the
  `phase2_benchmark.batched_thetas` draws. The first draw is a silent pool with an
  exactly zero gradient, so it would not benchmark a firing network. Default vs
  drawn timings agree within the IQR except the GPU gradient (1.7 % slower with
  default theta).
- "float64" runs are mixed precision (above). A pure float64 graph would need
  `PoolTwin` cast changes, which the brief rules out.

## Differences from the brief and roadmap

- Gradient checkpointing (`remat_segment`) is an addition beyond the brief: opt-in,
  default off, so every brief result above uses the plain scan.
- Native Ubuntu workstation instead of WSL2 on the laptop (no Windows driver
  steps apply).
- Theta rule for K = 1 (above).
- Panel A shows time per call (the measured quantity) for both sweeps and both
  operations; throughput, speed-up and efficiency are in the table. The G4
  cosines get their own panel (E).
- The G2 panel shows every variant and coordinate for four runs (precision x
  device) instead of two stacked bars, because the float32 result depends on the
  device.
- The brief cites float32 G2 and `b_config` G1 evidence that is no longer in the
  tree (trimmed in `5a4fa8db`); it is recoverable from `322cf9b5` and these runs
  reproduce it on CPU.
- `docs/DIGITAL_TWIN_ROADMAP.md` was removed in `5a4fa8db`
  (`docs/DIGITAL_TWIN_ROADMAP_REVISED.md` remains).
