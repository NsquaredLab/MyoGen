r"""
Spike Train Agreement Computation
==================================

Computes the rate of agreement between decomposed spike trains and ground truth.

Metrics computed:
- True Positives (TP): Ground truth spikes matched with decomposed spikes (±5ms tolerance)
- False Positives (FP): Decomposed spikes not matched with any ground truth spike
- False Negatives (FN): Ground truth spikes not matched with any decomposed spike
- Sensitivity/Recall: TP / (TP + FN) - proportion of ground truth spikes detected
- Precision: TP / (TP + FP) - proportion of decomposed spikes that are correct
- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)

For surface EMG: Compares decomp.pkl spike_trains with spike_trains.pkl
For intramuscular EMG: Compares decomp.pkl spike_trains with XML file

Usage:
------
python examples/synthetic_gen/11_compute_spike_train_agreement.py \
    --decomp-file results/synthetic_gen/semg_mu_41_42_43_plus18_snr10/decomp.pkl \
    --tolerance-ms 5.0
"""

import argparse
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.optimize import linear_sum_assignment

##############################################################################
# Configuration
##############################################################################

DEFAULT_TOLERANCE_MS = 5.0  # ±5ms tolerance for spike matching


##############################################################################
# Data Loading Functions
##############################################################################


def load_decomposition(decomp_path):
    """
    Load decomposition data from pickle file.

    Parameters
    ----------
    decomp_path : Path
        Path to decomp.pkl file.

    Returns
    -------
    dict
        Decomposition data dictionary.
    """
    if not decomp_path.exists():
        raise FileNotFoundError(f"Decomposition file not found: {decomp_path}")

    try:
        decomp = joblib.load(decomp_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load decomposition file: {e}")

    # Validate required keys
    required_keys = ["spike_trains", "mu_indices"]
    missing_keys = [key for key in required_keys if key not in decomp]
    if missing_keys:
        raise ValueError(f"Decomposition file missing required keys: {missing_keys}")

    return decomp


def load_ground_truth_spike_trains(decomp_folder):
    """
    Load ground truth spike trains from spike_trains.pkl (for surface EMG).

    Parameters
    ----------
    decomp_folder : Path
        Folder containing decomp.pkl.

    Returns
    -------
    list or None
        List of spike train arrays (in milliseconds), or None if not found.
    """
    spike_trains_path = decomp_folder / "spike_trains.pkl"

    if not spike_trains_path.exists():
        return None

    try:
        spike_trains = joblib.load(spike_trains_path)
        return spike_trains
    except Exception as e:
        print(f"⚠️  Failed to load {spike_trains_path}: {e}")
        return None


def load_spike_times_from_xml(xml_path):
    """
    Load spike timing data from emglab XML file (for intramuscular EMG).

    Parameters
    ----------
    xml_path : Path
        Path to XML file containing spike timing data.

    Returns
    -------
    dict
        Dictionary mapping motor unit IDs (int) to lists of spike times (float, in seconds).
        Example: {0: [0.5, 1.2, 1.8], 1: [0.7, 1.5, 2.1], ...}
    """
    if not xml_path.exists():
        return None

    spike_dict = {}

    try:
        with open(xml_path, 'r') as f:
            content = f.read()

        # Find the emglab_spike_events section
        start_tag = '<emglab_spike_events>'
        end_tag = '</emglab_spike_events>'

        start_idx = content.find(start_tag)
        end_idx = content.find(end_tag)

        if start_idx == -1 or end_idx == -1:
            return None

        # Extract the data section
        data_section = content[start_idx + len(start_tag):end_idx].strip()

        # Parse each line: time unit chan
        for line in data_section.split('\n'):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            time_s = float(parts[0])
            unit_id = int(parts[1])

            if unit_id not in spike_dict:
                spike_dict[unit_id] = []
            spike_dict[unit_id].append(time_s)

    except Exception as e:
        print(f"⚠️  Failed to parse XML file: {e}")
        return None

    # Sort spike times for each unit and convert to milliseconds
    for unit_id in spike_dict:
        spike_dict[unit_id] = sorted([t * 1000.0 for t in spike_dict[unit_id]])

    return spike_dict


##############################################################################
# Spike Train Agreement Computation
##############################################################################


def compute_spike_train_agreement(ground_truth_ms, decomposed_ms, tolerance_ms=5.0):
    """
    Compute agreement metrics between ground truth and decomposed spike trains.

    Uses a tolerance window of ±tolerance_ms to match spikes. A decomposed spike
    is considered a true positive if it falls within ±tolerance_ms of a ground
    truth spike.

    Parameters
    ----------
    ground_truth_ms : array-like
        Ground truth spike times in milliseconds.
    decomposed_ms : array-like
        Decomposed spike times in milliseconds.
    tolerance_ms : float, optional
        Tolerance window for spike matching in milliseconds, by default 5.0.

    Returns
    -------
    dict
        Dictionary containing:
        - 'tp': Number of true positives
        - 'fp': Number of false positives
        - 'fn': Number of false negatives
        - 'sensitivity': Sensitivity/Recall (TP / (TP + FN))
        - 'precision': Precision (TP / (TP + FP))
        - 'f1_score': F1 score
        - 'n_ground_truth': Number of ground truth spikes
        - 'n_decomposed': Number of decomposed spikes
    """
    gt = np.asarray(ground_truth_ms)
    dec = np.asarray(decomposed_ms)

    # Handle empty arrays
    if len(gt) == 0 and len(dec) == 0:
        return {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'sensitivity': 1.0,  # No spikes to detect, perfect
            'precision': 1.0,    # No false positives
            'f1_score': 1.0,
            'n_ground_truth': 0,
            'n_decomposed': 0,
        }
    elif len(gt) == 0:
        return {
            'tp': 0,
            'fp': len(dec),
            'fn': 0,
            'sensitivity': np.nan,  # Undefined
            'precision': 0.0,
            'f1_score': 0.0,
            'n_ground_truth': 0,
            'n_decomposed': len(dec),
        }
    elif len(dec) == 0:
        return {
            'tp': 0,
            'fp': 0,
            'fn': len(gt),
            'sensitivity': 0.0,
            'precision': np.nan,  # Undefined
            'f1_score': 0.0,
            'n_ground_truth': len(gt),
            'n_decomposed': 0,
        }

    # Match decomposed spikes to ground truth spikes
    # For each decomposed spike, find the closest ground truth spike
    gt_matched = np.zeros(len(gt), dtype=bool)
    dec_matched = np.zeros(len(dec), dtype=bool)

    for i, dec_spike in enumerate(dec):
        # Find ground truth spikes within tolerance window
        distances = np.abs(gt - dec_spike)
        within_tolerance = distances <= tolerance_ms

        if np.any(within_tolerance):
            # Find closest ground truth spike within tolerance
            closest_idx = np.argmin(distances)
            if distances[closest_idx] <= tolerance_ms and not gt_matched[closest_idx]:
                # Match this decomposed spike with the closest ground truth spike
                gt_matched[closest_idx] = True
                dec_matched[i] = True

    # Compute metrics
    tp = np.sum(gt_matched)  # Ground truth spikes that were detected
    fn = len(gt) - tp  # Ground truth spikes that were missed
    fp = np.sum(~dec_matched)  # Decomposed spikes that don't match any ground truth

    # Sensitivity (Recall): proportion of ground truth spikes detected
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Precision: proportion of decomposed spikes that are correct
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # F1 score: harmonic mean of precision and recall
    if precision + sensitivity > 0:
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1_score = 0.0

    return {
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'sensitivity': sensitivity,
        'precision': precision,
        'f1_score': f1_score,
        'n_ground_truth': len(gt),
        'n_decomposed': len(dec),
    }


def compute_agreement_matrix(gt_spike_trains, decomposed_spike_trains, tolerance_ms=5.0):
    """
    Compute exhaustive agreement matrix between all pairs of spike trains.

    Parameters
    ----------
    gt_spike_trains : list
        List of ground truth spike train arrays (in milliseconds).
    decomposed_spike_trains : list
        List of decomposed spike train arrays (in milliseconds).
    tolerance_ms : float, optional
        Tolerance window for spike matching in milliseconds, by default 5.0.

    Returns
    -------
    np.ndarray
        Agreement matrix with shape (n_ground_truth, n_decomposed).
        Each element [i, j] contains the F1 score between ground truth unit i
        and decomposed unit j.
    dict
        Dictionary containing detailed metrics for each pair.
    """
    n_gt = len(gt_spike_trains)
    n_dec = len(decomposed_spike_trains)

    # Initialize agreement matrix (F1 scores)
    agreement_matrix = np.zeros((n_gt, n_dec))

    # Store detailed metrics for each pair
    detailed_metrics = {}

    # Compute agreement for all pairs
    for i, gt_train in enumerate(gt_spike_trains):
        for j, dec_train in enumerate(decomposed_spike_trains):
            metrics = compute_spike_train_agreement(gt_train, dec_train, tolerance_ms)
            agreement_matrix[i, j] = metrics['f1_score']
            detailed_metrics[(i, j)] = metrics

    return agreement_matrix, detailed_metrics


def find_optimal_assignment(agreement_matrix):
    """
    Find optimal assignment between ground truth and decomposed units using Hungarian algorithm.

    Parameters
    ----------
    agreement_matrix : np.ndarray
        Agreement matrix with shape (n_ground_truth, n_decomposed).
        Each element [i, j] contains the F1 score between ground truth unit i
        and decomposed unit j.

    Returns
    -------
    list of tuples
        List of (gt_idx, dec_idx, f1_score) tuples representing optimal assignments.
    list
        List of unmatched ground truth indices.
    list
        List of unmatched decomposed indices.
    """
    n_gt, n_dec = agreement_matrix.shape

    # Hungarian algorithm minimizes cost, so we use negative F1 scores
    cost_matrix = -agreement_matrix

    # Solve assignment problem
    gt_indices, dec_indices = linear_sum_assignment(cost_matrix)

    # Extract assignments with F1 scores
    assignments = []
    for gt_idx, dec_idx in zip(gt_indices, dec_indices):
        f1_score = agreement_matrix[gt_idx, dec_idx]
        assignments.append((int(gt_idx), int(dec_idx), float(f1_score)))

    # Find unmatched units
    matched_gt = set(gt_indices)
    matched_dec = set(dec_indices)

    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_dec = [j for j in range(n_dec) if j not in matched_dec]

    return assignments, unmatched_gt, unmatched_dec


def plot_confusion_matrix(agreement_matrix, gt_labels, dec_labels, assignments, output_path):
    """
    Plot confusion matrix (agreement heatmap) with optimal assignments highlighted.

    Parameters
    ----------
    agreement_matrix : np.ndarray
        Agreement matrix with F1 scores.
    gt_labels : list
        Labels for ground truth units (rows).
    dec_labels : list
        Labels for decomposed units (columns).
    assignments : list of tuples
        Optimal assignments as (gt_idx, dec_idx, f1_score) tuples.
    output_path : Path
        Output file path for saving the plot.
    """
    # Set up matplotlib style
    plt.style.use('default')
    sns.set_context("paper", font_scale=1.2)

    # Create figure (wider aspect ratio for better readability)
    fig, ax = plt.subplots(figsize=(max(14, len(dec_labels) * 1.2),
                                     max(6, len(gt_labels) * 0.5)))

    # Create custom white-to-green colormap (white 0-0.5, then gradient to green)
    colors = ['#FFFFFF', '#FFFFFF', '#2ca02c', '#0d5e0d']  # white -> white -> medium green -> dark green
    positions = [0.0, 0.5, 0.75, 1.0]  # Stay white until 0.5, then gradient
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('white_green', list(zip(positions, colors)), N=n_bins)

    # Plot heatmap with F1 scores
    im = ax.imshow(agreement_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('F1 Score', rotation=270, labelpad=20)

    # Create plot-specific labels (1-indexed integers for visualization)
    gt_labels_plot = [str(i + 1) for i in range(len(gt_labels))]
    dec_labels_plot = [str(i + 1) for i in range(len(dec_labels))]

    # Set ticks and labels
    ax.set_xticks(np.arange(len(dec_labels)))
    ax.set_yticks(np.arange(len(gt_labels)))
    ax.set_xticklabels(dec_labels_plot, rotation=45, ha='right')
    ax.set_yticklabels(gt_labels_plot)

    # Labels
    ax.set_xlabel('Decomposed MU #', fontsize=12)
    ax.set_ylabel('Ground truth MU #', fontsize=12)
    ax.set_title('Spike Train Agreement Matrix (F1 Scores)', fontsize=14, pad=20)

    # Annotate cells with F1 scores using luminance-based text color
    for i in range(len(gt_labels)):
        for j in range(len(dec_labels)):
            f1 = agreement_matrix[i, j]
            # Get RGB color from colormap for this F1 value
            rgba = cmap(f1)
            # Calculate relative luminance (perceived brightness)
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            # Use black text on light backgrounds, white text on dark backgrounds
            text_color = 'black' if luminance > 0.5 else 'white'
            ax.text(j, i, f'{f1:.2f}', ha='center', va='center',
                   color=text_color, fontsize=9)

    # Highlight optimal assignments with boxes
    for gt_idx, dec_idx, _ in assignments:
        rect = plt.Rectangle((dec_idx - 0.5, gt_idx - 0.5), 1, 1,
                            fill=False, edgecolor='black', linewidth=3)
        ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


##############################################################################
# Main Processing Functions
##############################################################################


def process_surface_emg_agreement(decomp_path, tolerance_ms):
    """
    Process surface EMG spike train agreement with exhaustive matching.

    Compares decomp.pkl spike_trains with spike_trains.pkl using optimal assignment.

    Parameters
    ----------
    decomp_path : Path
        Path to decomp.pkl file.
    tolerance_ms : float
        Tolerance window for spike matching in milliseconds.

    Returns
    -------
    tuple
        (DataFrame with optimal assignment metrics, agreement_matrix, assignments, gt_labels, dec_labels)
        or None if data not found.
    """
    decomp_folder = decomp_path.parent

    # Load data
    # NOTE: For surface EMG, decomp.pkl contains ground truth (with MU indices)
    # and spike_trains.pkl contains decomposed results
    decomp = load_decomposition(decomp_path)
    gt_spike_trains = decomp['spike_trains']  # Ground truth with MU indices
    mu_indices = decomp['mu_indices']

    # Load decomposed spike trains
    decomp_spike_trains = load_ground_truth_spike_trains(decomp_folder)

    if decomp_spike_trains is None:
        print("⚠️  spike_trains.pkl not found - skipping surface EMG agreement")
        return None

    print(f"\n  Ground truth units: {len(gt_spike_trains)}")
    print(f"  Decomposed units: {len(decomp_spike_trains)}")

    # Compute exhaustive agreement matrix
    print(f"  Computing {len(gt_spike_trains)} x {len(decomp_spike_trains)} agreement matrix...")
    agreement_matrix, detailed_metrics = compute_agreement_matrix(
        gt_spike_trains, decomp_spike_trains, tolerance_ms
    )

    # Find optimal assignment
    assignments, unmatched_gt, unmatched_dec = find_optimal_assignment(agreement_matrix)

    print(f"  Optimal assignments: {len(assignments)}")
    print(f"  Unmatched ground truth: {len(unmatched_gt)}")
    print(f"  Unmatched decomposed: {len(unmatched_dec)}")

    # Build results for matched units
    results = []
    for gt_idx, dec_idx, f1_score in assignments:
        metrics = detailed_metrics[(gt_idx, dec_idx)]

        # Ground truth uses generic GT labels
        gt_label = f"GT_{gt_idx}"

        # Decomposed uses actual motor unit indices (to match MUAP plot names)
        # Decomposed spike trains correspond to mu_indices
        mu_idx = mu_indices[dec_idx] if dec_idx < len(mu_indices) else dec_idx
        dec_label = f"MU_{mu_idx}"

        results.append({
            'ground_truth_index': gt_idx,
            'decomposed_index': dec_idx,
            'motor_unit_index': mu_idx,
            'ground_truth_label': gt_label,
            'decomposed_label': dec_label,
            'true_positives': metrics['tp'],
            'false_positives': metrics['fp'],
            'false_negatives': metrics['fn'],
            'sensitivity_recall': metrics['sensitivity'],
            'precision': metrics['precision'],
            'f1_score': metrics['f1_score'],
            'n_ground_truth_spikes': metrics['n_ground_truth'],
            'n_decomposed_spikes': metrics['n_decomposed'],
            'matched': True,
        })

    # Add unmatched ground truth units (all spikes are false negatives)
    for gt_idx in unmatched_gt:
        n_spikes = len(gt_spike_trains[gt_idx])
        results.append({
            'ground_truth_index': gt_idx,
            'decomposed_index': -1,
            'motor_unit_index': -1,
            'ground_truth_label': f"GT_{gt_idx}",
            'decomposed_label': "UNMATCHED",
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': n_spikes,
            'sensitivity_recall': 0.0,
            'precision': np.nan,
            'f1_score': 0.0,
            'n_ground_truth_spikes': n_spikes,
            'n_decomposed_spikes': 0,
            'matched': False,
        })

    # Add unmatched decomposed units (all spikes are false positives)
    for dec_idx in unmatched_dec:
        mu_idx = mu_indices[dec_idx] if dec_idx < len(mu_indices) else dec_idx
        n_spikes = len(decomp_spike_trains[dec_idx])
        results.append({
            'ground_truth_index': -1,
            'decomposed_index': dec_idx,
            'motor_unit_index': mu_idx,
            'ground_truth_label': "UNMATCHED",
            'decomposed_label': f"MU_{mu_idx}",
            'true_positives': 0,
            'false_positives': n_spikes,
            'false_negatives': 0,
            'sensitivity_recall': np.nan,
            'precision': 0.0,
            'f1_score': 0.0,
            'n_ground_truth_spikes': 0,
            'n_decomposed_spikes': n_spikes,
            'matched': False,
        })

    # Create labels for confusion matrix
    # Ground truth: MU indices from decomp.pkl (rows)
    # Decomposed: generic DEC labels from spike_trains.pkl (columns)
    gt_labels = [f"MU_{mu_indices[i]}" if i < len(mu_indices) else f"GT_{i}"
                 for i in range(len(gt_spike_trains))]
    dec_labels = [f"DEC_{i}" for i in range(len(decomp_spike_trains))]

    return pd.DataFrame(results), agreement_matrix, assignments, gt_labels, dec_labels


def process_intramuscular_emg_agreement(decomp_path, tolerance_ms):
    """
    Process intramuscular EMG spike train agreement with exhaustive matching.

    Compares decomp.pkl spike_trains with XML file using optimal assignment.

    Parameters
    ----------
    decomp_path : Path
        Path to decomp.pkl file.
    tolerance_ms : float
        Tolerance window for spike matching in milliseconds.

    Returns
    -------
    tuple
        (DataFrame with optimal assignment metrics, agreement_matrix, assignments, gt_labels, dec_labels)
        or None if data not found.
    """
    decomp_folder = decomp_path.parent

    # Load decomposition data
    decomp = load_decomposition(decomp_path)
    gt_spike_trains = decomp['spike_trains']  # Ground truth from decomp.pkl
    mu_indices = decomp['mu_indices']

    # Load decomposed spike trains from XML
    xml_path = decomp_folder / f"{decomp_folder.name}.xml"
    decomposed_spike_dict = load_spike_times_from_xml(xml_path)

    if decomposed_spike_dict is None:
        print(f"⚠️  XML file not found or invalid: {xml_path.name} - skipping intramuscular EMG agreement")
        return None

    # Convert XML dict to list (sorted by XML unit ID)
    xml_unit_ids = sorted(decomposed_spike_dict.keys())
    decomposed_spike_trains = [decomposed_spike_dict[uid] for uid in xml_unit_ids]

    print(f"\n  Ground truth units: {len(gt_spike_trains)} (MU indices: {mu_indices})")
    print(f"  Decomposed units: {len(decomposed_spike_trains)} (XML IDs: {xml_unit_ids})")

    # Compute exhaustive agreement matrix
    print(f"  Computing {len(gt_spike_trains)} x {len(decomposed_spike_trains)} agreement matrix...")
    agreement_matrix, detailed_metrics = compute_agreement_matrix(
        gt_spike_trains, decomposed_spike_trains, tolerance_ms
    )

    # Find optimal assignment
    assignments, unmatched_gt, unmatched_dec = find_optimal_assignment(agreement_matrix)

    print(f"  Optimal assignments: {len(assignments)}")
    print(f"  Unmatched ground truth: {len(unmatched_gt)}")
    print(f"  Unmatched decomposed: {len(unmatched_dec)}")

    # Build results for matched units
    results = []
    for gt_idx, dec_idx, f1_score in assignments:
        metrics = detailed_metrics[(gt_idx, dec_idx)]

        # Get original labels
        mu_idx = mu_indices[gt_idx]
        xml_unit_id = xml_unit_ids[dec_idx]

        results.append({
            'ground_truth_index': gt_idx,
            'decomposed_index': dec_idx,
            'motor_unit_index': mu_idx,
            'xml_unit_id': xml_unit_id,
            'ground_truth_label': f"MU_{mu_idx}",
            'decomposed_label': f"XML_{xml_unit_id}",
            'true_positives': metrics['tp'],
            'false_positives': metrics['fp'],
            'false_negatives': metrics['fn'],
            'sensitivity_recall': metrics['sensitivity'],
            'precision': metrics['precision'],
            'f1_score': metrics['f1_score'],
            'n_ground_truth_spikes': metrics['n_ground_truth'],
            'n_decomposed_spikes': metrics['n_decomposed'],
            'matched': True,
        })

    # Add unmatched ground truth units (all spikes are false negatives)
    for gt_idx in unmatched_gt:
        mu_idx = mu_indices[gt_idx]
        n_spikes = len(gt_spike_trains[gt_idx])
        results.append({
            'ground_truth_index': gt_idx,
            'decomposed_index': -1,
            'motor_unit_index': mu_idx,
            'xml_unit_id': -1,
            'ground_truth_label': f"MU_{mu_idx}",
            'decomposed_label': "UNMATCHED",
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': n_spikes,
            'sensitivity_recall': 0.0,
            'precision': np.nan,
            'f1_score': 0.0,
            'n_ground_truth_spikes': n_spikes,
            'n_decomposed_spikes': 0,
            'matched': False,
        })

    # Add unmatched decomposed units (all spikes are false positives)
    for dec_idx in unmatched_dec:
        xml_unit_id = xml_unit_ids[dec_idx]
        n_spikes = len(decomposed_spike_trains[dec_idx])
        results.append({
            'ground_truth_index': -1,
            'decomposed_index': dec_idx,
            'motor_unit_index': -1,
            'xml_unit_id': xml_unit_id,
            'ground_truth_label': "UNMATCHED",
            'decomposed_label': f"XML_{xml_unit_id}",
            'true_positives': 0,
            'false_positives': n_spikes,
            'false_negatives': 0,
            'sensitivity_recall': np.nan,
            'precision': 0.0,
            'f1_score': 0.0,
            'n_ground_truth_spikes': 0,
            'n_decomposed_spikes': n_spikes,
            'matched': False,
        })

    # Create labels for confusion matrix
    gt_labels = [f"MU_{mu_indices[i]}" for i in range(len(gt_spike_trains))]
    dec_labels = [f"XML_{xml_unit_ids[i]}" for i in range(len(decomposed_spike_trains))]

    return pd.DataFrame(results), agreement_matrix, assignments, gt_labels, dec_labels


##############################################################################
# Main Function
##############################################################################


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute spike train agreement between decomposed and ground truth"
    )
    parser.add_argument(
        "--decomp-file",
        type=Path,
        required=True,
        help="Path to decomp.pkl file",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=DEFAULT_TOLERANCE_MS,
        help=f"Tolerance window for spike matching in milliseconds (default: {DEFAULT_TOLERANCE_MS})",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Spike Train Agreement Computation")
    print("=" * 80)
    print(f"Input: {args.decomp_file}")
    print(f"Tolerance: ±{args.tolerance_ms} ms")

    decomp_folder = args.decomp_file.parent
    folder_name = decomp_folder.name

    # Determine EMG type from folder name
    is_surface = folder_name.startswith('semg_')
    is_intramuscular = folder_name.startswith('iemg_')

    if not is_surface and not is_intramuscular:
        print("\n⚠️  Could not determine EMG type from folder name")
        print("    Expected folder name to start with 'semg_' or 'iemg_'")
        return

    print(f"EMG type: {'Surface EMG' if is_surface else 'Intramuscular EMG'}")

    # Process agreement based on EMG type
    if is_surface:
        print("\n📊 Computing surface EMG spike train agreement with exhaustive matching...")
        print("    Comparing: decomp.pkl vs spike_trains.pkl")
        result = process_surface_emg_agreement(args.decomp_file, args.tolerance_ms)
    else:
        print("\n📊 Computing intramuscular EMG spike train agreement with exhaustive matching...")
        print("    Comparing: decomp.pkl (ground truth) vs XML file (decomposed)")
        result = process_intramuscular_emg_agreement(args.decomp_file, args.tolerance_ms)

    if result is None:
        print("\n❌ No agreement data could be computed")
        return

    df_results, agreement_matrix, assignments, gt_labels, dec_labels = result

    # Save results to CSV
    output_path = decomp_folder / "spike_train_agreement.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n✅ Saved optimal assignment results to: {output_path}")

    # Save full agreement matrix to CSV
    matrix_output_path = decomp_folder / "spike_train_agreement_matrix.csv"
    df_matrix = pd.DataFrame(agreement_matrix, index=gt_labels, columns=dec_labels)
    df_matrix.to_csv(matrix_output_path)
    print(f"✅ Saved agreement matrix to: {matrix_output_path}")

    # Plot confusion matrix
    confusion_matrix_path = decomp_folder / "spike_train_confusion_matrix.png"
    plot_confusion_matrix(agreement_matrix, gt_labels, dec_labels, assignments, confusion_matrix_path)
    print(f"✅ Saved confusion matrix plot to: {confusion_matrix_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    # Separate matched and unmatched units
    df_matched = df_results[df_results['matched'] == True]
    df_unmatched = df_results[df_results['matched'] == False]

    print(f"\nTotal units:")
    print(f"  Matched pairs: {len(df_matched)}")
    print(f"  Unmatched: {len(df_unmatched)}")

    # Metrics for matched units only
    if len(df_matched) > 0:
        print(f"\nMatched units - Average metrics:")
        # Use nanmean to handle any NaN values
        print(f"  Sensitivity (Recall): {np.nanmean(df_matched['sensitivity_recall']):.3f} ± {np.nanstd(df_matched['sensitivity_recall']):.3f}")
        print(f"  Precision:            {np.nanmean(df_matched['precision']):.3f} ± {np.nanstd(df_matched['precision']):.3f}")
        print(f"  F1 Score:             {np.nanmean(df_matched['f1_score']):.3f} ± {np.nanstd(df_matched['f1_score']):.3f}")

    # Total spike counts
    total_tp = df_results['true_positives'].sum()
    total_fp = df_results['false_positives'].sum()
    total_fn = df_results['false_negatives'].sum()
    total_gt = df_results['n_ground_truth_spikes'].sum()
    total_dec = df_results['n_decomposed_spikes'].sum()

    print(f"\nTotal spikes:")
    print(f"  Ground truth: {total_gt}")
    print(f"  Decomposed:   {total_dec}")
    print(f"  True Positives:  {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")

    # Overall sensitivity and precision
    overall_sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_sensitivity) / (overall_precision + overall_sensitivity) if (overall_precision + overall_sensitivity) > 0 else 0.0

    print(f"\nOverall metrics (all spikes combined):")
    print(f"  Sensitivity (Recall): {overall_sensitivity:.3f}")
    print(f"  Precision:            {overall_precision:.3f}")
    print(f"  F1 Score:             {overall_f1:.3f}")

    # Print optimal assignments
    print(f"\n" + "=" * 80)
    print("Optimal Assignments")
    print("=" * 80)

    for _, row in df_matched.iterrows():
        gt_label = row['ground_truth_label']
        dec_label = row['decomposed_label']
        f1 = row['f1_score']
        print(f"  {gt_label} → {dec_label}  (F1: {f1:.3f}, Sens: {row['sensitivity_recall']:.3f}, Prec: {row['precision']:.3f})")

    if len(df_unmatched) > 0:
        print(f"\nUnmatched units:")
        for _, row in df_unmatched.iterrows():
            if row['ground_truth_index'] == -1:
                print(f"  {row['decomposed_label']}: No ground truth match (all {row['false_positives']} spikes are FP)")
            else:
                print(f"  {row['ground_truth_label']}: No decomposed match (all {row['false_negatives']} spikes are FN)")

    print("\n" + "=" * 80)
    print("✅ Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
