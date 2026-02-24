"""Statistical and visual comparison of YOLOv8 and Gemini classification results.

Loads per-simulation metrics for both models, generates box plots and line
charts, runs a paired Wilcoxon signed-rank test, and writes a summary report.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    COMPARISON_DIR,
    COMPARISON_PLOTS_DIR,
    GEMINI_METRICS_FILE,
    METRICS,
    METRIC_LABELS,
    NUM_SIMULATIONS,
    ROBOFLOW_METRICS_FILE,
    WILCOXON_ALPHA,
)

log = logging.getLogger(__name__)

METRIC_DISPLAY = dict(zip(METRICS, METRIC_LABELS))

sns.set_style("whitegrid")
plt.rcParams.update({"figure.figsize": (12, 8), "font.size": 11})


def load_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and align metrics CSVs for both models."""
    for path, name in [(ROBOFLOW_METRICS_FILE, "YOLOv8"), (GEMINI_METRICS_FILE, "Gemini")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} metrics not found: {path}")

    df_yolo = pd.read_csv(ROBOFLOW_METRICS_FILE).sort_values("simulation_number").reset_index(drop=True)
    df_gemini = pd.read_csv(GEMINI_METRICS_FILE).sort_values("simulation_number").reset_index(drop=True)

    if len(df_yolo) != len(df_gemini):
        common = set(df_yolo["simulation_number"]) & set(df_gemini["simulation_number"])
        df_yolo = df_yolo[df_yolo["simulation_number"].isin(common)].reset_index(drop=True)
        df_gemini = df_gemini[df_gemini["simulation_number"].isin(common)].reset_index(drop=True)
        log.warning("Simulation count mismatch; using %d common simulations", len(common))

    log.info("Loaded %d simulations for each model", len(df_yolo))
    return df_yolo, df_gemini


def plot_individual_boxplots(df_yolo: pd.DataFrame, df_gemini: pd.DataFrame) -> None:
    """One box plot per metric comparing both models."""
    for metric in METRICS:
        label = METRIC_DISPLAY[metric]
        fig, ax = plt.subplots(figsize=(10, 7))

        bp = ax.boxplot(
            [df_yolo[metric], df_gemini[metric]],
            labels=["YOLOv8", "Gemini"],
            patch_artist=True, notch=True, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="red", markeredgecolor="red", markersize=8),
        )
        for patch, color in zip(bp["boxes"], ["#a8c8e8", "#a8e8c0"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for i, vals in enumerate([df_yolo[metric], df_gemini[metric]], start=1):
            ax.scatter(np.random.normal(i, 0.04, size=len(vals)), vals, alpha=0.4, s=30, color="navy")

        ax.set_ylabel(label)
        ax.set_title(f"{label} — YOLOv8 vs Gemini ({len(df_yolo)} trials)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(COMPARISON_PLOTS_DIR / f"boxplot_{metric}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    log.info("Individual box plots saved")


def plot_combined_boxplot(df_yolo: pd.DataFrame, df_gemini: pd.DataFrame) -> None:
    """Side-by-side box plots for all metrics in one figure."""
    fig, ax = plt.subplots(figsize=(14, 8))
    pos_yolo = [1, 3, 5, 7]
    pos_gemini = [1.8, 3.8, 5.8, 7.8]

    bp1 = ax.boxplot([df_yolo[m] for m in METRICS], positions=pos_yolo, widths=0.6,
                     patch_artist=True, notch=True, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="red", markeredgecolor="red", markersize=8))
    bp2 = ax.boxplot([df_gemini[m] for m in METRICS], positions=pos_gemini, widths=0.6,
                     patch_artist=True, notch=True, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="red", markeredgecolor="red", markersize=8))

    for patch in bp1["boxes"]:
        patch.set_facecolor("#a8c8e8"); patch.set_alpha(0.7)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#a8e8c0"); patch.set_alpha(0.7)

    ax.set_xticks([1.4, 3.4, 5.4, 7.4])
    ax.set_xticklabels([METRIC_DISPLAY[m] for m in METRICS])
    ax.set_ylabel("Score")
    ax.set_title(f"All Metrics — YOLOv8 vs Gemini ({len(df_yolo)} trials)")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc="#a8c8e8", alpha=0.7, label="YOLOv8"),
        plt.Rectangle((0, 0), 1, 1, fc="#a8e8c0", alpha=0.7, label="Gemini"),
    ], loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(COMPARISON_PLOTS_DIR / "boxplot_all_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Combined box plot saved")


def plot_line_charts(df_yolo: pd.DataFrame, df_gemini: pd.DataFrame) -> None:
    """Per-simulation line charts for all metrics."""
    sims = df_yolo["simulation_number"]

    # Accuracy + F1 (two-panel)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    for ax, metric in zip([ax1, ax2], ["accuracy", "f1_score"]):
        label = METRIC_DISPLAY[metric]
        ax.plot(sims, df_yolo[metric], "o-", label="YOLOv8", color="blue", markersize=5, alpha=0.7)
        ax.plot(sims, df_gemini[metric], "s-", label="Gemini", color="green", markersize=5, alpha=0.7)
        ax.axhline(df_yolo[metric].mean(), color="blue", ls="--", alpha=0.4)
        ax.axhline(df_gemini[metric].mean(), color="green", ls="--", alpha=0.4)
        ax.set_ylabel(label)
        ax.set_title(f"{label} per Trial")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    ax2.set_xlabel("Trial")
    plt.tight_layout()
    fig.savefig(COMPARISON_PLOTS_DIR / "line_accuracy_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # All metrics (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, metric in zip(axes.flat, METRICS):
        label = METRIC_DISPLAY[metric]
        ax.plot(sims, df_yolo[metric], "o-", label="YOLOv8", color="blue", markersize=4, alpha=0.7)
        ax.plot(sims, df_gemini[metric], "s-", label="Gemini", color="green", markersize=4, alpha=0.7)
        ax.axhline(df_yolo[metric].mean(), color="blue", ls="--", alpha=0.4)
        ax.axhline(df_gemini[metric].mean(), color="green", ls="--", alpha=0.4)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle("All Metrics — YOLOv8 vs Gemini", fontweight="bold", y=1.0)
    plt.tight_layout()
    fig.savefig(COMPARISON_PLOTS_DIR / "line_all_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Line charts saved")


def run_wilcoxon(df_yolo: pd.DataFrame, df_gemini: pd.DataFrame) -> dict:
    """Paired Wilcoxon signed-rank test for each metric. Returns results dict."""
    results = {}
    for metric in METRICS:
        yolo_vals = df_yolo[metric].values
        gemini_vals = df_gemini[metric].values
        stat, p = sp_stats.wilcoxon(yolo_vals, gemini_vals)
        diff = float((yolo_vals - gemini_vals).mean())
        significant = p < WILCOXON_ALPHA

        results[metric] = {
            "label": METRIC_DISPLAY[metric],
            "statistic": float(stat),
            "p_value": float(p),
            "significant": significant,
            "mean_diff_yolo_minus_gemini": diff,
            "yolo_mean": float(yolo_vals.mean()),
            "gemini_mean": float(gemini_vals.mean()),
        }

        status = "SIG" if significant else "n.s."
        log.info("%-12s  W=%.2f  p=%.4f  diff=%+.4f  [%s]", METRIC_DISPLAY[metric], stat, p, diff, status)

    return results


def plot_wilcoxon(results: dict) -> None:
    """Bar chart of mean differences coloured by significance."""
    labels = [METRIC_DISPLAY[m] for m in METRICS]
    diffs = [results[m]["mean_diff_yolo_minus_gemini"] for m in METRICS]
    p_vals = [results[m]["p_value"] for m in METRICS]
    colors = ["#d9534f" if p < WILCOXON_ALPHA else "#999999" for p in p_vals]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(range(len(labels)), diffs, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", lw=1)

    for bar, p in zip(bars, p_vals):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"p={p:.4f}",
                ha="center", va="bottom" if h >= 0 else "top", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Difference (YOLOv8 − Gemini)")
    ax.set_title(f"Wilcoxon Signed-Rank Test (α = {WILCOXON_ALPHA})")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc="#d9534f", alpha=0.7, label=f"Significant (p < {WILCOXON_ALPHA})"),
        plt.Rectangle((0, 0), 1, 1, fc="#999999", alpha=0.7, label="Not significant"),
    ], loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(COMPARISON_PLOTS_DIR / "wilcoxon_results.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wilcoxon plot saved")


def save_report(df_yolo: pd.DataFrame, df_gemini: pd.DataFrame, wilcoxon: dict) -> None:
    """Write JSON results and a plain-text comparison report."""
    # JSON
    json_path = COMPARISON_PLOTS_DIR / "wilcoxon_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "test": "Wilcoxon Signed-Rank Test",
            "alpha": WILCOXON_ALPHA,
            "n_simulations": len(df_yolo),
            "timestamp": datetime.now().isoformat(),
            "results": wilcoxon,
        }, f, indent=2)

    # Text report
    report_path = COMPARISON_DIR / "comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Comparison Report — YOLOv8 vs Gemini\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"Simulations: {len(df_yolo)}\n\n")

        f.write("Descriptive Statistics\n")
        f.write("-" * 60 + "\n")
        for m in METRICS:
            label = METRIC_DISPLAY[m]
            f.write(f"{label}:\n")
            f.write(f"  YOLOv8  {df_yolo[m].mean():.4f} +/- {df_yolo[m].std():.4f}\n")
            f.write(f"  Gemini  {df_gemini[m].mean():.4f} +/- {df_gemini[m].std():.4f}\n\n")

        f.write("Wilcoxon Signed-Rank Test\n")
        f.write("-" * 60 + "\n")
        for m in METRICS:
            r = wilcoxon[m]
            sig = "significant" if r["significant"] else "not significant"
            f.write(f"{r['label']}: W={r['statistic']:.2f}, p={r['p_value']:.4f} ({sig})\n")

        yolo_wins = sum(1 for m in METRICS if wilcoxon[m]["significant"] and wilcoxon[m]["mean_diff_yolo_minus_gemini"] > 0)
        gemini_wins = sum(1 for m in METRICS if wilcoxon[m]["significant"] and wilcoxon[m]["mean_diff_yolo_minus_gemini"] < 0)
        f.write(f"\nSignificant wins — YOLOv8: {yolo_wins}, Gemini: {gemini_wins}, "
                f"No difference: {len(METRICS) - yolo_wins - gemini_wins}\n")

    log.info("Report saved to %s", report_path)


def run_comparison() -> None:
    """Full comparison pipeline."""
    COMPARISON_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df_yolo, df_gemini = load_metrics()
    plot_individual_boxplots(df_yolo, df_gemini)
    plot_combined_boxplot(df_yolo, df_gemini)
    plot_line_charts(df_yolo, df_gemini)
    wilcoxon = run_wilcoxon(df_yolo, df_gemini)
    plot_wilcoxon(wilcoxon)
    save_report(df_yolo, df_gemini, wilcoxon)
    log.info("Comparison complete. Results in %s", COMPARISON_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_comparison()