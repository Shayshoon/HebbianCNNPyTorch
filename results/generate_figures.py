"""Generate the frozen quantitative results figures from the CSV source tables.

Run from anywhere:
    python results/generate_figures.py

The script writes PNG (300 dpi), SVG, and PDF versions of every figure to
``results/figures``. It intentionally contains validation assertions for the
frozen totals and the corrected one-batch BP/A2 ratio.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator


COLORS = {
    "A1": "#0072B2",
    "A2": "#E69F00",
    "B": "#009E73",
    "BP": "#4D4D4D",
    "HB": "#0072B2",
}
FORMATS = ("png", "svg", "pdf")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def method_key(label: str) -> str:
    if label.startswith("A1"):
        return "A1"
    if label.startswith("A2"):
        return "A2"
    if label.startswith("BP"):
        return "BP"
    if label.startswith("HB"):
        return "HB"
    return "B"


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for extension in FORMATS:
        metadata = {"Title": stem}
        if extension in {"png", "pdf"}:
            metadata["Author"] = "Frozen quantitative audit"
        kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "facecolor": "white",
            "metadata": metadata,
        }
        if extension == "png":
            kwargs["dpi"] = 300
        fig.savefig(figures_dir / f"{stem}.{extension}", **kwargs)
    plt.close(fig)


def annotate_vertical_bars(ax: plt.Axes, bars, values, fmt) -> None:
    ceiling = max(values)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ceiling * 0.025,
            fmt(value),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#222222",
        )


def apply_conditional_hatch(bars, labels: list[str]) -> None:
    for bar, label in zip(bars, labels, strict=True):
        if method_key(label) == "A2":
            bar.set_hatch("///")
            bar.set_edgecolor("#7A5200")
            bar.set_linewidth(0.9)


def figure_same_work(data_dir: Path, figures_dir: Path) -> None:
    rows = read_csv(data_dir / "compute_same_work.csv")
    scopes = [
        ("one_batch", "One batch", 1e9, "B operations"),
        ("equal_20_epochs", "Equal 20-epoch scope", 1e12, "T operations"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    for ax, (scope, title, divisor, ylabel) in zip(axes, scopes, strict=True):
        selected = [r for r in rows if r["scope"] == scope]
        labels = [r["method_label"] for r in selected]
        values = [int(r["total_scalar_operations"]) / divisor for r in selected]
        colors = [COLORS[method_key(label)] for label in labels]
        bars = ax.bar(labels, values, color=colors, width=0.68)
        apply_conditional_hatch(bars, labels)
        annotate_vertical_bars(ax, bars, values, lambda x: f"{x:,.2f}")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.18)
        ax.tick_params(axis="x", rotation=18)
        style_axes(ax)
    one_batch = {r["method"]: int(r["total_scalar_operations"]) for r in rows if r["scope"] == "one_batch"}
    ratio = one_batch["BP"] / one_batch["A2"]
    assert round(ratio, 3) == 17.699, ratio
    fig.suptitle("Training-core compute at matched work", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.025,
        f"One-batch BP / A2 conditional = {ratio:.3f}×. A2 is a conditional sparse-aware estimate.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save_figure(fig, figures_dir, "fig01_compute_same_work")


def figure_training_budgets(data_dir: Path, figures_dir: Path) -> None:
    rows = read_csv(data_dir / "compute_training_budgets.csv")
    panels = [
        ("configured", "Configured budgets", "Hebbian 20 epochs vs BP 100 epochs"),
        ("convergence_sensitivity", "Convergence sensitivity", "Hebbian ≈5 epochs vs BP 100 epochs"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    for ax, (comparison, title, subtitle) in zip(axes, panels, strict=True):
        selected = [r for r in rows if r["comparison"] == comparison]
        labels = [f'{r["method_label"]} ({r["epochs"]} ep)' for r in selected]
        values = [int(r["total_scalar_operations"]) / 1e12 for r in selected]
        colors = [COLORS[method_key(r["method_label"])] for r in selected]
        y = list(range(len(selected)))
        bars = ax.barh(y, values, color=colors, height=0.62)
        apply_conditional_hatch(bars, [r["method_label"] for r in selected])
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(2.5, 1800)
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_xlabel("Total training-core operations (trillions, log scale)")
        ax.set_title(f"{title}\n{subtitle}", loc="left", fontweight="bold", fontsize=11)
        ax.grid(axis="x", which="both", color="#D9D9D9", linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        for bar, value in zip(bars, values, strict=True):
            ax.text(value * 1.08, bar.get_y() + bar.get_height() / 2, f"{value:,.2f} T", va="center", fontsize=8.5)
    fig.suptitle("Compute under configured and convergence-sensitivity budgets", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.02,
        "Neither panel is an equal-accuracy comparison. The ≈5-epoch Hebbian case is a stabilization sensitivity only.",
        ha="center",
        fontsize=9,
        color="#8A2D1C",
    )
    save_figure(fig, figures_dir, "fig02_compute_training_budgets")


def figure_method_level(data_dir: Path, figures_dir: Path) -> None:
    rows = read_csv(data_dir / "compute_method_level.csv")
    labels = [r["method_label"] for r in rows]
    values = [int(r["total_scalar_operations"]) / 1e12 for r in rows]
    colors = [COLORS[method_key(label)] for label in labels]
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.23)
    y = list(range(len(rows)))
    bars = ax.barh(y, values, color=colors, height=0.62)
    apply_conditional_hatch(bars, labels)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(25, 1600)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlabel("Full method-level operations (trillions, log scale)")
    ax.set_title("Full method-level compute", loc="left", fontsize=15, fontweight="bold")
    ax.grid(axis="x", which="both", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(value * 1.06, bar.get_y() + bar.get_height() / 2, f"{value:,.2f} T", va="center", fontsize=9)
    fig.text(
        0.5,
        0.035,
        "Includes shared workflow costs. ZCA eigendecomposition sensitivity is non-exact and changes Hebbian totals by 0.019%–0.820%.",
        ha="center",
        fontsize=8.7,
        color="#444444",
    )
    save_figure(fig, figures_dir, "fig03_compute_method_level")


def figure_persistent_memory(data_dir: Path, figures_dir: Path) -> None:
    rows = read_csv(data_dir / "memory_persistent_state.csv")
    labels = [r["method_label"] for r in rows]
    values = [float(r["persistent_megabytes"]) for r in rows]
    colors = [COLORS[method_key(label)] for label in labels]
    fig, ax = plt.subplots(figsize=(8.7, 4.9))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.25)
    bars = ax.bar(labels, values, color=colors, width=0.68)
    apply_conditional_hatch(bars, labels)
    annotate_vertical_bars(ax, bars, values, lambda x: f"{x:.3f} MB")
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_ylabel("Persistent state (MB)")
    ax.set_title("Persistent training state", loc="left", fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", rotation=13)
    style_axes(ax)
    fig.text(0.5, 0.035, "A2 is conditional and uses the CSR int32 storage estimate.", ha="center", fontsize=9, color="#444444")
    save_figure(fig, figures_dir, "fig04_memory_persistent_state")


def figure_peak_memory(data_dir: Path, figures_dir: Path) -> None:
    training = read_csv(data_dir / "memory_peak_training.csv")
    zca = read_csv(data_dir / "memory_zca_construction.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 6.6), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=0.25, right=0.98, top=0.86, bottom=0.16, wspace=0.62)

    labels = [f'{r["method_label"]} — {r["scope"]}' for r in training]
    values = [float(r["peak_megabytes"]) for r in training]
    colors = [COLORS[method_key(r["method_label"])] for r in training]
    y = list(range(len(training)))
    bars = ax1.barh(y, values, color=colors, height=0.58)
    apply_conditional_hatch(bars, [r["method_label"] for r in training])
    ax1.set_yticks(y, labels, fontsize=8.2)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 320)
    ax1.set_xlabel("Peak memory (MB)")
    ax1.set_title("Per-training reference scopes", loc="left", fontweight="bold")
    ax1.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax1.text(value + 4, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)

    zlabels = [r["stage_label"] for r in zca]
    zvalues = [float(r["peak_megabytes"]) for r in zca]
    zy = list(range(len(zca)))
    zcolors = ["#8C6BB1", "#8073AC", "#542788", "#B2ABD2"]
    zbars = ax2.barh(zy, zvalues, color=zcolors, height=0.58)
    ax2.set_yticks(zy, zlabels, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 420)
    ax2.set_xlabel("Memory (MB)")
    ax2.set_title("One-time ZCA construction", loc="left", fontweight="bold")
    ax2.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(zbars, zvalues, strict=True):
        ax2.text(value + 5, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)

    fig.suptitle("Peak-memory references: training and ZCA construction", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.025,
        "BP markers are alternative reference points (baseline, one-buffer, two-buffer), not additive components. ZCA construction is shown separately.",
        ha="center",
        fontsize=8.7,
        color="#444444",
    )
    save_figure(fig, figures_dir, "fig05_memory_peak_and_zca")


def figure_movement(data_dir: Path, figures_dir: Path) -> None:
    rows = read_csv(data_dir / "logical_tensor_movement.csv")
    labels = [r["method_label"] for r in rows]
    values = [float(r["logical_movement_megabytes"]) for r in rows]
    colors = [COLORS[method_key(label)] for label in labels]
    fig, ax = plt.subplots(figsize=(8.7, 4.9))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.25)
    bars = ax.bar(labels, values, color=colors, width=0.68)
    apply_conditional_hatch(bars, labels)
    annotate_vertical_bars(ax, bars, values, lambda x: f"{x:.3f} MB")
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel("Logical tensor movement (MB)")
    ax.set_title("Logical tensor movement", loc="left", fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", rotation=12)
    style_axes(ax)
    fig.text(0.5, 0.035, "This is an analytical movement estimate, not peak memory or measured hardware bandwidth.", ha="center", fontsize=9, color="#444444")
    save_figure(fig, figures_dir, "fig06_logical_tensor_movement")


def figure_accuracy(data_dir: Path, figures_dir: Path) -> None:
    rows = read_csv(data_dir / "accuracy_context.csv")
    exact = [r for r in rows if r["evidence_type"] == "exact"]
    approximate = [r for r in rows if r["evidence_type"] == "approximate_graph_read"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.9))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.25, wspace=0.14)

    exact_labels = ["Original paper\nTable 1", "Saved notebook\nsingle run"]
    exact_values = [float(r["accuracy_percent"]) for r in exact]
    exact_err = [float(r["uncertainty_percent"]) if r["uncertainty_percent"] else 0 for r in exact]
    exact_colors = ["#0072B2", "#56B4E9"]
    for x, value, error, color in zip(range(2), exact_values, exact_err, exact_colors, strict=True):
        ax1.errorbar(x, value, yerr=error if error else None, fmt="o", markersize=9, capsize=4, color=color, linewidth=1.5)
        suffix = f" ± {error:.1f}" if error else ""
        ax1.text(x, value + 0.28, f"{value:.2f}%{suffix}", ha="center", fontsize=9)
    ax1.set_xticks([0, 1], exact_labels)
    ax1.set_xlim(-0.55, 1.55)
    ax1.set_ylim(63.7, 65.35)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Exact reported values", loc="left", fontweight="bold")
    style_axes(ax1)

    marker_map = {"HB": "o", "BP": "s"}
    offset_map = {"HB": -1.4, "BP": 1.4}
    for row in approximate:
        method = row["method"]
        epoch = int(row["epochs"])
        value = float(row["accuracy_percent"])
        x = epoch + offset_map[method]
        ax2.scatter(
            x,
            value,
            s=78,
            marker=marker_map[method],
            facecolors="white",
            edgecolors=COLORS[method],
            linewidths=1.8,
            label=method if epoch == 20 else None,
            zorder=3,
        )
        ax2.text(x, value + 0.65, f"≈{value:.0f}%", ha="center", fontsize=9, color=COLORS[method])
    ax2.set_xticks([20, 100], ["20 epochs", "100 epochs"])
    ax2.set_xlim(10, 110)
    ax2.set_ylim(54, 73)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Approximate comparison-paper graph reads", loc="left", fontweight="bold")
    ax2.legend(frameon=False, loc="lower right")
    style_axes(ax2)
    fig.suptitle("Accuracy evidence remains source- and protocol-specific", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.035,
        "No curve is inferred. Approximate points are hollow; the comparison paper's shaded SD over 10 runs is not numerically reproduced.",
        ha="center",
        fontsize=8.8,
        color="#444444",
    )
    save_figure(fig, figures_dir, "fig07_accuracy_context")


def validate_frozen_inputs(data_dir: Path) -> None:
    same_work = read_csv(data_dir / "compute_same_work.csv")
    frozen = {(r["scope"], r["method"]): int(r["total_scalar_operations"]) for r in same_work}
    assert frozen[("one_batch", "A1")] == 8_667_340_084
    assert frozen[("one_batch", "A2")] == 1_380_339_683
    assert frozen[("one_batch", "B")] == 17_267_771_387
    assert frozen[("one_batch", "BP")] == 24_430_482_934
    assert f'{frozen[("one_batch", "BP")] / frozen[("one_batch", "A2")]:.3f}' == "17.699"

    method_level = read_csv(data_dir / "compute_method_level.csv")
    values = {(r["method"], int(r["epochs"])): int(r["total_scalar_operations"]) for r in method_level}
    assert values[("A1", 20)] == 112_590_041_834_287
    assert values[("A2", 20)] == 35_365_252_130_627
    assert values[("B", 20)] == 198_609_206_182_487
    assert values[("BP", 20)] == 245_157_703_690_020
    assert values[("BP", 100)] == 1_222_377_021_050_100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    data_dir = results_root / "data"
    figures_dir = results_root / "figures"

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#111111",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 120,
            "savefig.pad_inches": 0.12,
        }
    )
    validate_frozen_inputs(data_dir)
    figure_same_work(data_dir, figures_dir)
    figure_training_budgets(data_dir, figures_dir)
    figure_method_level(data_dir, figures_dir)
    figure_persistent_memory(data_dir, figures_dir)
    figure_peak_memory(data_dir, figures_dir)
    figure_movement(data_dir, figures_dir)
    figure_accuracy(data_dir, figures_dir)
    print(f"Generated {len(FORMATS) * 7} files in {figures_dir}")


if __name__ == "__main__":
    main()
