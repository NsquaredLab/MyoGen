# Spike Train Agreement Analysis

This document describes the spike train agreement computation tools for evaluating decomposition quality.

## Overview

Two scripts have been created to compute the rate of agreement between decomposed spike trains and ground truth:

1. **`11_compute_spike_train_agreement.py`** - Computes agreement for a single directory with **exhaustive matching**
2. **`12_batch_compute_agreement.py`** - Batch processes multiple directories

## Methodology

### Exhaustive Matching Approach

The scripts use an **exhaustive matching algorithm** that:

1. **Computes all pairwise agreements** between ground truth and decomposed spike trains
2. **Creates an N×M agreement matrix** where N = # ground truth units, M = # decomposed units
3. **Finds optimal assignment** using the Hungarian algorithm (maximizes total F1 score)
4. **Handles unmatched units** (when N ≠ M)

This approach **solves the unit correspondence problem**: decomposition algorithms may identify correct motor units but number them differently than the ground truth.

### Metrics Computed

The scripts compute the following metrics using a ±5ms tolerance window for spike matching:

- **True Positives (TP)**: Ground truth spikes matched with decomposed spikes
- **False Positives (FP)**: Decomposed spikes not matched with any ground truth spike
- **False Negatives (FN)**: Ground truth spikes not matched with any decomposed spike
- **Sensitivity (Recall)**: TP / (TP + FN) - proportion of ground truth spikes detected
- **Precision**: TP / (TP + FP) - proportion of decomposed spikes that are correct
- **F1 Score**: Harmonic mean of precision and recall

### Data Sources

**For Surface EMG (semg_* directories):**
- **Ground Truth**: `spike_trains.pkl` - simulated spike trains
- **Decomposed**: `decomp.pkl['spike_trains']` - decomposed spike trains

**For Intramuscular EMG (iemg_* directories):**
- **Ground Truth**: `decomp.pkl['spike_trains']` - simulated spike trains
- **Decomposed**: `*.xml` file - decomposed spike trains from emglab

### Spike Matching Algorithm

**Step 1: Pairwise Agreement Matrix**
- For each (ground truth unit, decomposed unit) pair:
  - Compare spike trains with ±tolerance_ms window
  - Compute F1 score for that pair
- Result: N×M matrix of F1 scores

**Step 2: Optimal Assignment (Hungarian Algorithm)**
- Find assignment that maximizes total F1 score
- Ensures each ground truth unit matched to at most one decomposed unit
- Ensures each decomposed unit matched to at most one ground truth unit

**Step 3: Individual Spike Matching**
For each matched pair:
1. Find all decomposed spikes within ±tolerance_ms of ground truth spikes
2. Match to closest ground truth spike within tolerance
3. Ensure each ground truth spike matched at most once
4. Count matched spikes as TP, unmatched decomposed as FP, unmatched ground truth as FN

## Usage

### Single Directory Analysis

```bash
uv run python examples/synthetic_gen/11_compute_spike_train_agreement.py \
    --decomp-file results/synthetic_gen/semg_mu_41_42_43_plus18_snr10/decomp.pkl \
    --tolerance-ms 5.0
```

**Output:**
- `spike_train_agreement.csv` - Detailed per-motor-unit metrics
- Console output with summary statistics

### Batch Analysis

Process all directories matching a pattern:

```bash
# Process all directories with SNR=1
uv run python examples/synthetic_gen/12_batch_compute_agreement.py \
    --pattern "*snr1" \
    --tolerance-ms 5.0

# Process all surface EMG directories
uv run python examples/synthetic_gen/12_batch_compute_agreement.py \
    --pattern "semg_*" \
    --tolerance-ms 5.0

# Process all intramuscular EMG directories
uv run python examples/synthetic_gen/12_batch_compute_agreement.py \
    --pattern "iemg_*" \
    --tolerance-ms 5.0

# Process ALL directories (default)
uv run python examples/synthetic_gen/12_batch_compute_agreement.py \
    --tolerance-ms 5.0
```

## Output Files

Each processed directory will contain:

**`spike_train_agreement.csv`** - Optimal assignment results with columns:
- `ground_truth_index` - Index in ground truth spike train list
- `decomposed_index` - Index in decomposed spike train list
- `motor_unit_index` - Motor unit index from decomposed data
- `xml_unit_id` - XML unit ID (intramuscular only)
- `ground_truth_label` - Generic label for ground truth unit (GT_0, GT_1, ...)
- `decomposed_label` - Motor unit label matching MUAP plots (MU_41, MU_42, ...)
- `true_positives` - Number of correctly detected spikes
- `false_positives` - Number of false detections
- `false_negatives` - Number of missed spikes
- `sensitivity_recall` - TP / (TP + FN)
- `precision` - TP / (TP + FP)
- `f1_score` - Harmonic mean of precision and recall
- `n_ground_truth_spikes` - Total ground truth spikes
- `n_decomposed_spikes` - Total decomposed spikes
- `matched` - Boolean indicating if unit was matched

**`spike_train_agreement_matrix.csv`** - Full N×M agreement matrix
- Rows: Ground truth units (GT_0, GT_1, ...)
- Columns: Decomposed units (MU_41, MU_42, ... matching MUAP plot names)
- Values: F1 scores for all pairwise comparisons

**`spike_train_confusion_matrix.png`** - Visual heatmap
- Color-coded F1 scores for all pairs
- Blue boxes highlight optimal assignments

## Example Results

### Surface EMG (semg_mu_41_42_43_plus18_snr10)

```
Ground truth units: 9
Decomposed units: 21
Computing 9 x 21 agreement matrix...

Total units:
  Matched pairs: 9
  Unmatched: 12 (false positives)

Matched units - Average metrics:
  Sensitivity (Recall): 0.514 ± 0.347
  Precision:            0.517 ± 0.338
  F1 Score:             0.515 ± 0.343

Overall metrics (all spikes combined):
  Sensitivity (Recall): 0.510
  Precision:            0.222
  F1 Score:             0.310

Optimal Assignments:
  GT_0 → MU_55  (F1: 1.000)  ← Perfect match!
  GT_1 → MU_46  (F1: 0.384)
  GT_2 → MU_51  (F1: 0.291)
  ...
  GT_4 → MU_54  (F1: 1.000)  ← Perfect match!
  GT_8 → MU_41  (F1: 0.986)  ← Near perfect!

Unmatched: MU_42, MU_43, MU_47, MU_48, ... (12 false positive units)
```

**Key insight**: Three units show perfect/near-perfect agreement when properly matched!

### Intramuscular EMG (iemg_mu_76_77_78_79_80_snr1)

```
Ground truth units: 5 (MU indices: [76, 77, 78, 79, 80])
Decomposed units: 5 (XML IDs: [1, 2, 3, 4, 5])
Computing 5 x 5 agreement matrix...

Total units:
  Matched pairs: 5
  Unmatched: 0

Matched units - Average metrics:
  Sensitivity (Recall): 1.000 ± 0.000
  Precision:            1.000 ± 0.000
  F1 Score:             1.000 ± 0.000

Overall metrics (all spikes combined):
  Sensitivity (Recall): 1.000
  Precision:            1.000
  F1 Score:             1.000

Optimal Assignments:
  MU_76 → XML_5  (F1: 1.000)
  MU_77 → XML_4  (F1: 1.000)
  MU_78 → XML_3  (F1: 1.000)
  MU_79 → XML_1  (F1: 1.000)
  MU_80 → XML_2  (F1: 1.000)
```

**Key insight**: Perfect decomposition! The old index-based method gave F1=0.227 because XML numbering differed from ground truth. Exhaustive matching correctly identifies perfect agreement.

## Confusion Matrix Visualization

The confusion matrix heatmap provides a visual representation of the agreement between all ground truth and decomposed units:

- **Rows**: Ground truth motor units (labeled GT_0, GT_1, GT_2, ...)
- **Columns**: Decomposed motor units (labeled MU_41, MU_42, MU_43, ... to match MUAP plots)
- **Colors**: F1 scores (white = 0.0, dark red = 1.0)
- **Blue boxes**: Optimal assignments from Hungarian algorithm
- **Cell annotations**: F1 score values

### Reading the Confusion Matrix

- **Diagonal pattern**: If optimal assignments form a diagonal, unit numbering matches
- **Off-diagonal pattern**: Unit correspondence problem - numbering differs (expected!)
- **Dark red cells**: High agreement (F1 > 0.7)
- **Light yellow cells**: Poor agreement (F1 < 0.3)
- **Row with all low values**: Ground truth unit not properly decomposed
- **Column with all low values**: Decomposed unit is likely a false positive

**Example interpretation**: If you see GT_0 matched to MU_55 with F1=1.000, this means the ground truth unit at position 0 perfectly matches the decomposed motor unit 55 (which appears in MUAP plots as mu_055.svg).

## Interpreting Results

### Quality Thresholds

- **F1 Score ≥ 0.9**: Excellent agreement
- **F1 Score ≥ 0.7**: Good agreement
- **F1 Score ≥ 0.5**: Moderate agreement
- **F1 Score < 0.5**: Poor agreement

### Trade-offs

- **High Sensitivity, Low Precision**: Decomposition is over-detecting (many false positives)
- **Low Sensitivity, High Precision**: Decomposition is conservative (missing spikes)
- **Balanced Sensitivity & Precision**: Optimal decomposition quality

## Notes

1. The ±5ms tolerance window is standard for spike train comparison in EMG decomposition
2. Results depend heavily on signal quality (SNR), motor unit overlap, and decomposition algorithm
3. XML files may use different unit numbering (1, 2, 3...) than the original simulation
4. For surface EMG, mismatches between ground truth and decomposed spike train counts may indicate partial decomposition

## Technical Details

### Hungarian Algorithm

The optimal assignment problem is solved using the Hungarian algorithm (Kuhn-Munkres algorithm):
- **Input**: N×M cost matrix (negative F1 scores)
- **Output**: Optimal one-to-one assignment maximizing total F1
- **Complexity**: O(N³) for square matrices
- **Implementation**: `scipy.optimize.linear_sum_assignment`

The algorithm guarantees the optimal solution - no other assignment will have a higher total F1 score across all matched pairs.

### Handling Asymmetric Cases

When N ≠ M (different number of ground truth vs decomposed units):
- **N > M**: Some ground truth units remain unmatched (missed units)
- **N < M**: Some decomposed units remain unmatched (false positive units)
- Hungarian algorithm automatically handles this by matching min(N, M) pairs

## References

- **Spike train agreement metrics**: Negro et al., 2016, J Neurophysiol
  - Standard ±5ms tolerance window for spike matching
  - Sensitivity, precision, and F1 score definitions

- **Hungarian algorithm**: Kuhn, H.W., 1955, Naval Research Logistics Quarterly
  - Optimal assignment problem solution
  - Implemented in SciPy: `scipy.optimize.linear_sum_assignment`

- **EMG decomposition validation**: Holobar & Zazula, 2007, IEEE Trans Biomed Eng
  - DEMUSE methodology for decomposition quality assessment
