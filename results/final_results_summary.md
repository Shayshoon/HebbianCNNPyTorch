# Final quantitative results package

## Status

This package visualizes the frozen analytical methodology without changing the notebooks or any frozen derivation. Every plotted value is stored in a CSV file under `results/data/`, and every figure is provided as a 300 dpi PNG, editable SVG, and publication-ready PDF.

The corrected one-batch ratio is verified directly from the frozen totals:

`24,430,482,934 / 1,380,339,683 = 17.698891...`, reported as **17.699×**.

## Figure inventory

| Figure | Scope | Recommended use |
|---|---|---|
| `fig01_compute_same_work` | One-batch and equal-20-epoch training-core compute | Main report; concise presentation option |
| `fig02_compute_training_budgets` | Configured Hebbian-20/BP-100 budgets and Hebbian-≈5/BP-100 convergence sensitivity | Report; presentation only with the non-equivalence note visible |
| `fig03_compute_method_level` | Full method-level A1/A2/B at 20 epochs and BP at 20/100 epochs | Main report |
| `fig04_memory_persistent_state` | Persistent state for A1, A2 conditional, B, and BP+Adam | Report; presentation option |
| `fig05_memory_peak_and_zca` | Training peak-memory reference scopes and separately paneled one-time ZCA construction | Main report or appendix |
| `fig06_logical_tensor_movement` | Logical tensor movement as its own analytical resource metric | Report or appendix |
| `fig07_accuracy_context` | Exact reported values and approximate comparison-paper graph reads | Main report; presentation option |

Each stem above has `.png`, `.svg`, and `.pdf` variants in `results/figures/`.

## Source-data inventory

- `compute_same_work.csv`: one-batch and equal-20-epoch training-core totals.
- `compute_training_budgets.csv`: configured and convergence-sensitivity training budgets; `matched_accuracy` is explicitly `False`.
- `compute_method_level.csv`: full method-level totals.
- `memory_persistent_state.csv`: persistent state estimates.
- `memory_peak_training.csv`: training peak-memory references with exact scope labels.
- `memory_zca_construction.csv`: one-time ZCA construction and persistent ZCA matrix references.
- `logical_tensor_movement.csv`: analytical tensor-movement estimates, separate from memory.
- `accuracy_context.csv`: source location and evidence type for every accuracy point.
- `zca_eigendecomposition_sensitivity.csv`: non-exact ZCA eigendecomposition sensitivity range and resulting percentage impact.

## Interpretation guardrails preserved

- **A2 is conditional.** It is always labeled `A2 conditional` or otherwise identified as a conditional sparse-aware estimate.
- **The convergence panel is not an equal-accuracy claim.** Hebbian ≈5 epochs versus BP 100 epochs is a stabilization/convergence sensitivity only.
- **Accuracy evidence is not pooled.** The original-paper table value and the saved-notebook output are exact reported observations from distinct artifacts; the comparison-paper values are approximate graph reads.
- **No accuracy curve is reconstructed.** Only the available 20- and 100-epoch approximate points are shown, with no connecting line or invented intermediate epochs.
- **Peak-memory references are alternatives, not additive components.** The BP baseline, one-gradient-buffer, and conservative two-gradient-buffer values remain separately labeled.
- **ZCA construction is separate from per-training memory.** It occupies a distinct panel and source table.
- **Logical movement is not memory.** It is plotted in a separate figure and described as an analytical movement estimate, not measured bandwidth.
- **Method-level ZCA sensitivity remains non-exact.** The frozen 38,654,705,664–289,910,292,480 operation range changes Hebbian method totals by 0.019%–0.820%.

## Deliberate omission

A combined compute-versus-accuracy figure was not produced. The available compute budgets and accuracy observations are not aligned tightly enough by training protocol, source, run aggregation, and evidence type to support a defensible joint trade-off chart. Keeping them separate avoids implying an accuracy-matched comparison.

## Reproduction

From the repository root, run:

```powershell
python -m pip install -r results/requirements.txt
python results/generate_figures.py
```

The script reads only the CSV files in `results/data/`, validates the frozen headline totals (including 17.699×), and regenerates all 21 figure files in `results/figures/`.
