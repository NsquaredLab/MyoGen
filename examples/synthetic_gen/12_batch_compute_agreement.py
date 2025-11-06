r"""
Batch Spike Train Agreement Computation
========================================

Runs spike train agreement computation on all semg_* and iemg_* directories
in results/synthetic_gen/.

Usage:
------
python examples/synthetic_gen/12_batch_compute_agreement.py --tolerance-ms 5.0
"""

import argparse
from pathlib import Path
import subprocess
import sys

##############################################################################
# Configuration
##############################################################################

RESULTS_DIR = Path("results/synthetic_gen")
DEFAULT_TOLERANCE_MS = 5.0


##############################################################################
# Main Function
##############################################################################


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Batch compute spike train agreement for all directories"
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=DEFAULT_TOLERANCE_MS,
        help=f"Tolerance window for spike matching in milliseconds (default: {DEFAULT_TOLERANCE_MS})",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*emg_*",
        help="Glob pattern for directories to process (default: *emg_*)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Batch Spike Train Agreement Computation")
    print("=" * 80)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Directory pattern: {args.pattern}")
    print(f"Tolerance: ±{args.tolerance_ms} ms")

    if not RESULTS_DIR.exists():
        print(f"\n❌ Error: Results directory not found: {RESULTS_DIR}")
        return

    # Find all matching directories
    directories = sorted([d for d in RESULTS_DIR.glob(args.pattern) if d.is_dir()])

    if not directories:
        print(f"\n❌ No directories found matching pattern: {args.pattern}")
        return

    print(f"\nFound {len(directories)} directories to process:")
    for d in directories:
        print(f"  - {d.name}")

    # Process each directory
    print("\n" + "=" * 80)
    print("Processing directories...")
    print("=" * 80)

    success_count = 0
    skip_count = 0
    error_count = 0

    for i, directory in enumerate(directories, 1):
        decomp_file = directory / "decomp.pkl"

        print(f"\n[{i}/{len(directories)}] {directory.name}")

        if not decomp_file.exists():
            print(f"  ⚠️  Skipping: decomp.pkl not found")
            skip_count += 1
            continue

        # Run agreement computation
        cmd = [
            sys.executable,
            "examples/synthetic_gen/11_compute_spike_train_agreement.py",
            "--decomp-file", str(decomp_file),
            "--tolerance-ms", str(args.tolerance_ms),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                # Extract summary from output
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'Overall metrics' in line or 'Sensitivity' in line or 'Precision' in line or 'F1 Score' in line:
                        print(f"  {line.strip()}")
                success_count += 1
            else:
                print(f"  ❌ Error (exit code {result.returncode})")
                if result.stderr:
                    # Print first error line
                    error_lines = result.stderr.strip().split('\n')
                    print(f"     {error_lines[0]}")
                error_count += 1

        except subprocess.TimeoutExpired:
            print(f"  ❌ Error: timeout")
            error_count += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            error_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("Batch Processing Summary")
    print("=" * 80)
    print(f"Total directories: {len(directories)}")
    print(f"  ✅ Success: {success_count}")
    print(f"  ⚠️  Skipped: {skip_count}")
    print(f"  ❌ Errors:  {error_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
