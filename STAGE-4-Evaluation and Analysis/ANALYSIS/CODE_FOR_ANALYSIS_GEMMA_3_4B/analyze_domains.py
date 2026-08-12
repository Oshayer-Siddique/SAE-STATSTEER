#!/usr/bin/env python3
"""
Deep evaluation analysis for LOGIC, POLITIC, and SENTIMENT.

Run from the collected_results directory:
    CODE_FOR_ANALYSIS_GEMMA_2_2B/.venv/bin/python CODE_FOR_ANALYSIS_GEMMA_3_4B/analyze_domains.py

Outputs:
    CODE_FOR_ANALYSIS_GEMMA_3_4B/outputs_logic/
    CODE_FOR_ANALYSIS_GEMMA_3_4B/outputs_politic/
    CODE_FOR_ANALYSIS_GEMMA_3_4B/outputs_sentiment/
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "gemma_3_4B"
BASE_OUT = Path(__file__).resolve().parent
METRIC_KEYS = [
    "primary_score",
    "relevance_score",
    "richness_score",
    "coherence_score",
    "factuality_score",
]
FAILURE_KEYWORDS = [
    "repetitive",
    "repetition",
    "incoherent",
    "coherence",
    "off-topic",
    "irrelevant",
    "hallucinated",
    "fabricated",
    "false",
    "incorrect",
    "contradicts",
    "vague",
    "empty",
]


@dataclass(frozen=True)
class DomainSpec:
    key: str
    file_prefix: str
    title: str
    primary_name: str
    positive_shift_label: str
    score_meaning: str
    category_labels: tuple[str, str, str]
    positive_goal: str
    caveat: str


SPECS = {
    "logic": DomainSpec(
        key="logic",
        file_prefix="LOGIC",
        title="LOGIC",
        primary_name="logical correctness",
        positive_shift_label="correctness improvement",
        score_meaning="1 = contradicts ground truth / logically false, 5 = ambiguous, 10 = correct and logically sound",
        category_labels=("Incorrect", "Ambiguous", "Correct"),
        positive_goal="higher correctness",
        caveat="For LOGIC, higher primary score is directly better because it means closer agreement with the ground truth.",
    ),
    "politic": DomainSpec(
        key="politic",
        file_prefix="POLITIC",
        title="POLITIC",
        primary_name="political polarity",
        positive_shift_label="rightward polarity shift",
        score_meaning="1 = far left/progressive, 5 = neutral/centrist, 10 = far right/conservative",
        category_labels=("Left", "Center", "Right"),
        positive_goal="rightward shift",
        caveat="For POLITIC, higher primary score means more right-leaning, not automatically better. Interpret deltas as directional polarity shifts.",
    ),
    "sentiment": DomainSpec(
        key="sentiment",
        file_prefix="SENTIMENT",
        title="SENTIMENT",
        primary_name="sentiment polarity",
        positive_shift_label="positive sentiment shift",
        score_meaning="1 = very negative, 5 = neutral, 10 = very positive",
        category_labels=("Negative", "Neutral", "Positive"),
        positive_goal="positive sentiment shift",
        caveat="For SENTIMENT, higher primary score means more positive sentiment, not automatically better unless the steering target is positivity.",
    ),
}


def safe_float(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_or_nan(values: Iterable[Optional[float]]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def std_or_nan(values: Iterable[Optional[float]]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def ci95(values: Iterable[Optional[float]]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return float("nan")
    return 1.984 * std_or_nan(vals) / math.sqrt(len(vals))


def pct(value: float) -> str:
    return "NA" if value is None or math.isnan(value) else f"{100 * value:.1f}%"


def table_md(df: pd.DataFrame) -> str:
    text = df.astype(str)
    cols = list(text.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(row[c] for c in cols) + " |")
    return "\n".join(lines)


def score_category(score: Optional[float], labels: tuple[str, str, str]) -> str:
    if score is None or math.isnan(score):
        return "Unknown"
    if score <= 3:
        return labels[0]
    if score <= 6:
        return labels[1]
    return labels[2]


def parse_domain_files(spec: DomainSpec) -> pd.DataFrame:
    pattern = re.compile(
        rf"^{spec.file_prefix}_(?P<layer>\d+)_results_eval_(?P<alpha>[0-9.]+)\.json$"
    )
    rows = []
    for path in sorted(ROOT.glob(f"{spec.file_prefix}_*_results_eval_*.json")):
        match = pattern.match(path.name)
        if not match:
            continue
        layer = int(match.group("layer"))
        alpha = float(match.group("alpha"))
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for result in payload.get("results", []):
            sm = result.get("steered_metrics", {}) or {}
            bm = result.get("baseline_metrics", {}) or {}
            row = {
                "file": path.name,
                "domain": spec.key,
                "layer": layer,
                "alpha": alpha,
                "sample": result.get("sample"),
                "winner": result.get("winner", ""),
                "explanation": result.get("explanation", ""),
            }
            for key in METRIC_KEYS:
                metric = key.replace("_score", "")
                s_val = safe_float(sm.get(key))
                b_val = safe_float(bm.get(key))
                row[f"steered_{metric}"] = s_val
                row[f"baseline_{metric}"] = b_val
                row[f"delta_{metric}"] = (
                    s_val - b_val if s_val is not None and b_val is not None else float("nan")
                )
            row["json_diff"] = safe_float(result.get("diff"))
            row["primary_win"] = row["delta_primary"] > 0 if not math.isnan(row["delta_primary"]) else False
            row["primary_loss"] = row["delta_primary"] < 0 if not math.isnan(row["delta_primary"]) else False
            row["primary_tie"] = row["delta_primary"] == 0 if not math.isnan(row["delta_primary"]) else False
            row["steered_bad_output"] = bool(
                (row["steered_richness"] is not None and row["steered_richness"] <= 2)
                or (row["steered_coherence"] is not None and row["steered_coherence"] <= 2)
            )
            row["baseline_bad_output"] = bool(
                (row["baseline_richness"] is not None and row["baseline_richness"] <= 2)
                or (row["baseline_coherence"] is not None and row["baseline_coherence"] <= 2)
            )
            row["clean_directional_success"] = bool(
                row["delta_primary"] > 0
                and row["steered_relevance"] is not None
                and row["steered_richness"] is not None
                and row["steered_coherence"] is not None
                and row["steered_relevance"] >= 7
                and row["steered_richness"] >= 4
                and row["steered_coherence"] >= 4
            )
            quality = [row["delta_relevance"], row["delta_richness"], row["delta_coherence"]]
            if not math.isnan(row["delta_factuality"]):
                quality.append(row["delta_factuality"])
            row["delta_quality"] = mean_or_nan(quality)
            row["baseline_category"] = score_category(row["baseline_primary"], spec.category_labels)
            row["steered_category"] = score_category(row["steered_primary"], spec.category_labels)
            row["baseline_low"] = bool(row["baseline_primary"] is not None and row["baseline_primary"] <= 3)
            row["baseline_high"] = bool(row["baseline_primary"] is not None and row["baseline_primary"] >= 7)
            row["low_to_mid_or_high"] = bool(row["baseline_low"] and row["steered_primary"] is not None and row["steered_primary"] >= 5)
            row["low_to_high"] = bool(row["baseline_low"] and row["steered_primary"] is not None and row["steered_primary"] >= 7)
            row["high_to_below_mid"] = bool(row["baseline_high"] and row["steered_primary"] is not None and row["steered_primary"] < 5)
            row["extremity_delta"] = (
                abs(row["steered_primary"] - 5) - abs(row["baseline_primary"] - 5)
                if row["steered_primary"] is not None and row["baseline_primary"] is not None
                else float("nan")
            )
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No files found for {spec.title}")
    return pd.DataFrame(rows)


def summarize_configs(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (layer, alpha), g in samples.groupby(["layer", "alpha"], sort=True):
        primary_wins = g[g["primary_win"]]
        low = g[g["baseline_low"]]
        high = g[g["baseline_high"]]
        row = {
            "layer": layer,
            "alpha": alpha,
            "n": len(g),
            "primary_delta_mean": mean_or_nan(g["delta_primary"]),
            "primary_delta_median": float(np.nanmedian(g["delta_primary"])),
            "primary_delta_sd": std_or_nan(g["delta_primary"]),
            "primary_delta_ci95": ci95(g["delta_primary"]),
            "steered_primary_mean": mean_or_nan(g["steered_primary"]),
            "baseline_primary_mean": mean_or_nan(g["baseline_primary"]),
            "quality_delta_mean": mean_or_nan(g["delta_quality"]),
            "relevance_delta_mean": mean_or_nan(g["delta_relevance"]),
            "richness_delta_mean": mean_or_nan(g["delta_richness"]),
            "coherence_delta_mean": mean_or_nan(g["delta_coherence"]),
            "factuality_delta_mean": mean_or_nan(g["delta_factuality"]),
            "judge_steered_win_rate": float((g["winner"] == "Steered").mean()),
            "judge_baseline_win_rate": float((g["winner"] == "Baseline").mean()),
            "judge_same_rate": float((g["winner"] == "Same").mean()),
            "primary_win_rate": float(g["primary_win"].mean()),
            "primary_loss_rate": float(g["primary_loss"].mean()),
            "primary_tie_rate": float(g["primary_tie"].mean()),
            "clean_directional_success_rate": float(g["clean_directional_success"].mean()),
            "directional_success_bad_rate": (
                float(primary_wins["steered_bad_output"].mean()) if len(primary_wins) else float("nan")
            ),
            "baseline_low_n": len(low),
            "low_to_mid_or_high_rate": float(low["low_to_mid_or_high"].mean()) if len(low) else float("nan"),
            "low_to_high_rate": float(low["low_to_high"].mean()) if len(low) else float("nan"),
            "baseline_high_n": len(high),
            "high_to_below_mid_rate": float(high["high_to_below_mid"].mean()) if len(high) else float("nan"),
            "extremity_delta_mean": mean_or_nan(g["extremity_delta"]),
            "steered_bad_output_rate": float(g["steered_bad_output"].mean()),
            "baseline_bad_output_rate": float(g["baseline_bad_output"].mean()),
        }
        row["bad_output_rate_delta"] = row["steered_bad_output_rate"] - row["baseline_bad_output_rate"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["layer", "alpha"]).reset_index(drop=True)


def pivot(summary: pd.DataFrame, value: str) -> pd.DataFrame:
    table = summary.pivot(index="layer", columns="alpha", values=value)
    return table.sort_index().reindex(sorted(table.columns), axis=1)


def heatmap(out_dir: Path, summary: pd.DataFrame, value: str, title: str, name: str, cmap: str, center=False, fmt=".2f", cbar_label="") -> None:
    table = pivot(summary, value)
    data = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    if center:
        max_abs = np.nanmax(np.abs(data))
        vmin, vmax = -max_abs, max_abs
    else:
        vmin, vmax = np.nanmin(data), np.nanmax(data)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_xticklabels([f"{x:g}" for x in table.columns])
    ax.set_yticklabels([str(x) for x in table.index])
    ax.set_xlabel("Steering strength alpha")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    max_abs_data = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            label = "NA" if math.isnan(val) else format(val, fmt)
            if center and not math.isnan(val) and val > 0:
                label = f"+{label}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if not math.isnan(val) and abs(val) > 0.55 * max_abs_data else "black",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def alpha_response(out_dir: Path, spec: DomainSpec, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for layer, g in summary.groupby("layer", sort=True):
        g = g.sort_values("alpha")
        ax.plot(g["alpha"], g["primary_delta_mean"], marker="o", linewidth=2, label=f"Layer {layer}")
        ax.fill_between(
            g["alpha"],
            g["primary_delta_mean"] - g["primary_delta_ci95"],
            g["primary_delta_mean"] + g["primary_delta_ci95"],
            alpha=0.12,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Steering strength alpha")
    ax.set_ylabel(f"Mean {spec.primary_name} delta")
    ax.set_title(f"{spec.title}: Alpha Response by Layer")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{spec.key}_alpha_response_primary_delta.png", dpi=300)
    fig.savefig(out_dir / f"{spec.key}_alpha_response_primary_delta.pdf")
    plt.close(fig)


def pareto(out_dir: Path, spec: DomainSpec, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    layers = sorted(summary["layer"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(layers)))
    for layer, color in zip(layers, colors):
        g = summary[summary["layer"] == layer]
        ax.scatter(g["quality_delta_mean"], g["primary_delta_mean"], s=90, color=color, edgecolor="black", linewidth=0.6, label=f"Layer {layer}")
        for _, row in g.iterrows():
            ax.annotate(f"L{int(row.layer)}, a={row.alpha:g}", (row.quality_delta_mean, row.primary_delta_mean), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Mean quality delta")
    ax.set_ylabel(f"Mean {spec.primary_name} delta")
    ax.set_title(f"{spec.title}: Primary Shift vs Quality Preservation")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{spec.key}_pareto_primary_vs_quality.png", dpi=300)
    fig.savefig(out_dir / f"{spec.key}_pareto_primary_vs_quality.pdf")
    plt.close(fig)


def transition_table(samples: pd.DataFrame, labels: tuple[str, str, str]) -> pd.DataFrame:
    counts = pd.crosstab(samples["baseline_category"], samples["steered_category"])
    counts = counts.reindex(index=labels, columns=labels, fill_value=0)
    return counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)


def transition_heatmap(out_dir: Path, spec: DomainSpec, samples: pd.DataFrame, title: str, name: str) -> None:
    labels = spec.category_labels
    rates = transition_table(samples, labels)
    data = rates.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 5.3))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Steered category")
    ax.set_ylabel("Baseline category")
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, "NA" if math.isnan(val) else f"{100 * val:.1f}%", ha="center", va="center", color="white" if not math.isnan(val) and val > 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized transition rate")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def win_tie_loss(out_dir: Path, spec: DomainSpec, summary: pd.DataFrame) -> None:
    df = summary.sort_values(["layer", "alpha"])
    labels = [f"L{int(r.layer)}\na={r.alpha:g}" for r in df.itertuples()]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x, df["judge_steered_win_rate"], label="Steered", color="#2a9d8f")
    ax.bar(x, df["judge_same_rate"], bottom=df["judge_steered_win_rate"], label="Same", color="#e9c46a")
    ax.bar(x, df["judge_baseline_win_rate"], bottom=df["judge_steered_win_rate"] + df["judge_same_rate"], label="Baseline", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Judge outcome rate")
    ax.set_title(f"{spec.title}: Pairwise Judge Outcomes")
    ax.legend(frameon=False, ncol=3)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{spec.key}_judge_win_tie_loss_stacked.png", dpi=300)
    fig.savefig(out_dir / f"{spec.key}_judge_win_tie_loss_stacked.pdf")
    plt.close(fig)


def layer_robustness(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, g in summary.groupby("layer", sort=True):
        best = g.loc[g["primary_delta_mean"].idxmax()]
        worst = g.loc[g["primary_delta_mean"].idxmin()]
        rows.append({
            "layer": layer,
            "configs": len(g),
            "positive_configs": int((g["primary_delta_mean"] > 0).sum()),
            "positive_config_rate": float((g["primary_delta_mean"] > 0).mean()),
            "mean_primary_delta": mean_or_nan(g["primary_delta_mean"]),
            "sd_primary_delta_across_alphas": std_or_nan(g["primary_delta_mean"]),
            "range_primary_delta": float(g["primary_delta_mean"].max() - g["primary_delta_mean"].min()),
            "best_alpha": best.alpha,
            "best_primary_delta": best.primary_delta_mean,
            "worst_alpha": worst.alpha,
            "worst_primary_delta": worst.primary_delta_mean,
            "mean_quality_delta": mean_or_nan(g["quality_delta_mean"]),
            "mean_clean_directional_success": mean_or_nan(g["clean_directional_success_rate"]),
        })
    return pd.DataFrame(rows)


def alpha_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha, g in summary.groupby("alpha", sort=True):
        best = g.loc[g["primary_delta_mean"].idxmax()]
        rows.append({
            "alpha": alpha,
            "configs": len(g),
            "positive_configs": int((g["primary_delta_mean"] > 0).sum()),
            "positive_config_rate": float((g["primary_delta_mean"] > 0).mean()),
            "mean_primary_delta": mean_or_nan(g["primary_delta_mean"]),
            "mean_quality_delta": mean_or_nan(g["quality_delta_mean"]),
            "mean_clean_directional_success": mean_or_nan(g["clean_directional_success_rate"]),
            "best_layer": best.layer,
            "best_primary_delta": best.primary_delta_mean,
        })
    return pd.DataFrame(rows)


def prompt_steerability(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample, g in samples.groupby("sample", sort=True):
        rows.append({
            "sample": sample,
            "configs": len(g),
            "mean_primary_delta": mean_or_nan(g["delta_primary"]),
            "median_primary_delta": float(np.nanmedian(g["delta_primary"])),
            "positive_config_rate": float(g["primary_win"].mean()),
            "negative_config_rate": float(g["primary_loss"].mean()),
            "clean_directional_success_rate": float(g["clean_directional_success"].mean()),
            "mean_quality_delta": mean_or_nan(g["delta_quality"]),
        })
    return pd.DataFrame(rows).sort_values("mean_primary_delta", ascending=False)


def judge_agreement(samples: pd.DataFrame) -> pd.DataFrame:
    data = samples.copy()
    data["primary_relation"] = np.where(data["primary_win"], "Primary: Steered", np.where(data["primary_loss"], "Primary: Baseline", "Primary: Tie"))
    matrix = pd.crosstab(data["primary_relation"], data["winner"])
    return matrix.reindex(index=["Primary: Steered", "Primary: Tie", "Primary: Baseline"], columns=["Steered", "Same", "Baseline"], fill_value=0)


def failure_keywords(samples: pd.DataFrame) -> pd.DataFrame:
    explanations = samples["explanation"].fillna("").str.lower()
    rows = []
    for kw in FAILURE_KEYWORDS:
        count = int(explanations.str.contains(re.escape(kw)).sum())
        rows.append({"keyword": kw, "count": count, "rate": count / len(samples) if len(samples) else float("nan")})
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def fmt_summary_rows(df: pd.DataFrame, n=12) -> pd.DataFrame:
    d = df.sort_values("primary_delta_mean", ascending=False).head(n).copy()
    d.insert(0, "rank", range(1, len(d) + 1))
    return pd.DataFrame({
        "Rank": d["rank"],
        "Layer": d["layer"].astype(int),
        "Alpha": d["alpha"].map(lambda x: f"{x:g}"),
        "Steered Avg": d["steered_primary_mean"].map(lambda x: f"{x:.2f}"),
        "Baseline Avg": d["baseline_primary_mean"].map(lambda x: f"{x:.2f}"),
        "Delta": d["primary_delta_mean"].map(lambda x: f"{x:+.2f}"),
        "Primary Success": d["primary_win_rate"].map(pct),
        "Judge Win": d["judge_steered_win_rate"].map(pct),
        "Clean Success": d["clean_directional_success_rate"].map(pct),
        "Quality Delta": d["quality_delta_mean"].map(lambda x: f"{x:+.3f}"),
    })


def fmt_layer(layer_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Layer": layer_df["layer"].astype(int),
        "Positive Configs": layer_df.apply(lambda r: f"{int(r.positive_configs)}/{int(r.configs)}", axis=1),
        "Positive Rate": layer_df["positive_config_rate"].map(pct),
        "Mean Delta": layer_df["mean_primary_delta"].map(lambda x: f"{x:+.3f}"),
        "Best Alpha": layer_df["best_alpha"].map(lambda x: f"{x:g}"),
        "Best Delta": layer_df["best_primary_delta"].map(lambda x: f"{x:+.3f}"),
        "Volatility SD": layer_df["sd_primary_delta_across_alphas"].map(lambda x: f"{x:.3f}"),
        "Mean Quality Delta": layer_df["mean_quality_delta"].map(lambda x: f"{x:+.3f}"),
        "Mean Clean Success": layer_df["mean_clean_directional_success"].map(pct),
    })


def write_report(out_dir: Path, spec: DomainSpec, samples: pd.DataFrame, summary: pd.DataFrame, layer_df: pd.DataFrame, alpha_df: pd.DataFrame, prompt_df: pd.DataFrame, judge_df: pd.DataFrame, fail_df: pd.DataFrame) -> None:
    best = summary.loc[summary["primary_delta_mean"].idxmax()]
    positive_summary = summary[summary["primary_delta_mean"] > 0]
    quality_pool = positive_summary if not positive_summary.empty else summary
    best_quality = quality_pool.loc[quality_pool["quality_delta_mean"].idxmax()]
    clean_pool = positive_summary if not positive_summary.empty else summary
    best_clean = clean_pool.loc[clean_pool["clean_directional_success_rate"].idxmax()]
    worst = summary.loc[summary["primary_delta_mean"].idxmin()]
    positive = int((summary["primary_delta_mean"] > 0).sum())
    significant = summary[(summary["primary_delta_mean"] - summary["primary_delta_ci95"]) > 0]
    trans_all = transition_table(samples, spec.category_labels).copy()
    for col in trans_all.columns:
        trans_all[col] = trans_all[col].map(pct)
    trans_all = trans_all.reset_index().rename(columns={"baseline_category": "Baseline"})
    judge = judge_df.copy()
    judge["Total"] = judge.sum(axis=1)
    judge = judge.reset_index().rename(columns={"primary_relation": "Primary Relation"})
    top_clean = clean_pool.sort_values("clean_directional_success_rate", ascending=False).head(10)
    top_clean_fmt = pd.DataFrame({
        "Layer": top_clean["layer"].astype(int),
        "Alpha": top_clean["alpha"].map(lambda x: f"{x:g}"),
        "Clean Success": top_clean["clean_directional_success_rate"].map(pct),
        "Delta": top_clean["primary_delta_mean"].map(lambda x: f"{x:+.3f}"),
        "Quality Delta": top_clean["quality_delta_mean"].map(lambda x: f"{x:+.3f}"),
    })
    conversion = summary.sort_values("low_to_mid_or_high_rate", ascending=False).head(10)
    conv_fmt = pd.DataFrame({
        "Layer": conversion["layer"].astype(int),
        "Alpha": conversion["alpha"].map(lambda x: f"{x:g}"),
        f"{spec.category_labels[0]} Baseline N": conversion["baseline_low_n"].astype(int),
        f"{spec.category_labels[0]} -> >=5": conversion["low_to_mid_or_high_rate"].map(pct),
        f"{spec.category_labels[0]} -> {spec.category_labels[2]}": conversion["low_to_high_rate"].map(pct),
        "Delta": conversion["primary_delta_mean"].map(lambda x: f"{x:+.3f}"),
    })
    regression = summary.sort_values("high_to_below_mid_rate", ascending=False).head(10)
    reg_fmt = pd.DataFrame({
        "Layer": regression["layer"].astype(int),
        "Alpha": regression["alpha"].map(lambda x: f"{x:g}"),
        f"{spec.category_labels[2]} Baseline N": regression["baseline_high_n"].astype(int),
        f"{spec.category_labels[2]} -> <5": regression["high_to_below_mid_rate"].map(pct),
        "Delta": regression["primary_delta_mean"].map(lambda x: f"{x:+.3f}"),
    })
    alpha_fmt = pd.DataFrame({
        "Alpha": alpha_df["alpha"].map(lambda x: f"{x:g}"),
        "Positive Configs": alpha_df.apply(lambda r: f"{int(r.positive_configs)}/{int(r.configs)}", axis=1),
        "Mean Delta": alpha_df["mean_primary_delta"].map(lambda x: f"{x:+.3f}"),
        "Mean Quality Delta": alpha_df["mean_quality_delta"].map(lambda x: f"{x:+.3f}"),
        "Mean Clean Success": alpha_df["mean_clean_directional_success"].map(pct),
        "Best Layer": alpha_df["best_layer"].astype(int),
        "Best Delta": alpha_df["best_primary_delta"].map(lambda x: f"{x:+.3f}"),
    })
    prompt_top = prompt_df.head(8)
    prompt_bottom = prompt_df.tail(8).sort_values("mean_primary_delta")
    prompt_top_fmt = pd.DataFrame({
        "Sample": prompt_top["sample"].astype(int),
        "Mean Delta": prompt_top["mean_primary_delta"].map(lambda x: f"{x:+.3f}"),
        "Positive Config Rate": prompt_top["positive_config_rate"].map(pct),
        "Clean Success": prompt_top["clean_directional_success_rate"].map(pct),
        "Quality Delta": prompt_top["mean_quality_delta"].map(lambda x: f"{x:+.3f}"),
    })
    prompt_bottom_fmt = pd.DataFrame({
        "Sample": prompt_bottom["sample"].astype(int),
        "Mean Delta": prompt_bottom["mean_primary_delta"].map(lambda x: f"{x:+.3f}"),
        "Positive Config Rate": prompt_bottom["positive_config_rate"].map(pct),
        "Clean Success": prompt_bottom["clean_directional_success_rate"].map(pct),
        "Quality Delta": prompt_bottom["mean_quality_delta"].map(lambda x: f"{x:+.3f}"),
    })
    fail_fmt = fail_df.head(10).copy()
    fail_fmt = pd.DataFrame({"Keyword": fail_fmt["keyword"], "Count": fail_fmt["count"].astype(int), "Rate": fail_fmt["rate"].map(pct)})

    improvement_word = "improvement" if spec.key == "logic" else "positive-direction shift"
    report = f"""# {spec.title} Steering Evaluation Summary

This report analyzes `{spec.file_prefix}_<layer>_results_eval_<alpha>.json`.

## Metric Meaning

Primary score meaning:

```text
{spec.score_meaning}
```

Important caveat:

> {spec.caveat}

The main effect size is:

```text
Primary delta = mean(steered primary_score - baseline primary_score)
```

For this domain, positive delta means `{spec.positive_shift_label}`.

## Overall Result

Configurations analyzed: `{len(summary)}`

Samples per configuration: `{int(summary['n'].iloc[0])}`

Positive-delta configurations:

```text
{positive} / {len(summary)} = {100 * positive / len(summary):.1f}%
```

Mean config-level delta: `{summary['primary_delta_mean'].mean():+.3f}`

Median config-level delta: `{summary['primary_delta_mean'].median():+.3f}`

Configurations with approximate 95% CI excluding zero positively: `{len(significant)}`

## Executive Interpretation

The {spec.title} results should be read as a layer-alpha steering profile, not a single universal success result. The best raw setting is:

```text
Layer {int(best.layer)}, alpha {best.alpha:g}, delta {best.primary_delta_mean:+.3f}
```

The best quality-preserving setting is:

```text
Layer {int(best_quality.layer)}, alpha {best_quality.alpha:g}, quality delta {best_quality.quality_delta_mean:+.3f}, primary delta {best_quality.primary_delta_mean:+.3f}
```

The best strict clean-success setting is:

```text
Layer {int(best_clean.layer)}, alpha {best_clean.alpha:g}, clean success {pct(best_clean.clean_directional_success_rate)}
```

The worst setting is:

```text
Layer {int(worst.layer)}, alpha {worst.alpha:g}, delta {worst.primary_delta_mean:+.3f}
```

## Ranked Top Settings

{table_md(fmt_summary_rows(summary, n=min(16, len(summary))))}

## Best Raw Setting

| Metric | Value |
| --- | ---: |
| Layer | {int(best.layer)} |
| Alpha | {best.alpha:g} |
| Steered primary mean | {best.steered_primary_mean:.2f} |
| Baseline primary mean | {best.baseline_primary_mean:.2f} |
| Primary delta | {best.primary_delta_mean:+.3f} |
| 95% CI | +/- {best.primary_delta_ci95:.3f} |
| Primary success ratio | {pct(best.primary_win_rate)} |
| Judge win ratio | {pct(best.judge_steered_win_rate)} |
| Clean directional success | {pct(best.clean_directional_success_rate)} |
| Quality delta | {best.quality_delta_mean:+.3f} |

Interpretation: this setting gives the strongest raw {spec.primary_name} shift. For LOGIC this can be discussed as correctness improvement. For POLITIC and SENTIMENT it should be discussed as directional polarity movement rather than automatic quality improvement.

## Best Quality-Preserving Setting

| Metric | Value |
| --- | ---: |
| Layer | {int(best_quality.layer)} |
| Alpha | {best_quality.alpha:g} |
| Primary delta | {best_quality.primary_delta_mean:+.3f} |
| Quality delta | {best_quality.quality_delta_mean:+.3f} |
| Judge win ratio | {pct(best_quality.judge_steered_win_rate)} |
| Clean directional success | {pct(best_quality.clean_directional_success_rate)} |
| Steered bad-output rate | {pct(best_quality.steered_bad_output_rate)} |
| Baseline bad-output rate | {pct(best_quality.baseline_bad_output_rate)} |

Interpretation: this setting is useful when the goal is not only directional shift but preserving relevance, richness, and coherence.

## Clean Directional Success

Definition:

```text
delta_primary > 0
AND steered_relevance >= 7
AND steered_richness >= 4
AND steered_coherence >= 4
```

Top settings:

{table_md(top_clean_fmt)}

Interpretation: clean success is stricter than raw success. It measures whether steering produces a usable directional win rather than only a score shift.

## Low-to-High Conversion

This asks whether steering rescues low-scoring baseline outputs.

{table_md(conv_fmt)}

For LOGIC, this means incorrect-to-ambiguous/correct or incorrect-to-correct conversion. For SENTIMENT, it means negative-to-neutral/positive movement. For POLITIC, it means left-to-center/right movement, which is directional rather than inherently better.

## High-to-Low Regression

This asks whether steering damages high-scoring baseline outputs.

{table_md(reg_fmt)}

For LOGIC, this is a correctness regression. For SENTIMENT and POLITIC, this is movement away from the high end of the polarity scale.

## Category Transition Matrix

Categories:

```text
{spec.category_labels[0]}: 1-3
{spec.category_labels[1]}: 4-6
{spec.category_labels[2]}: 7-10
```

{table_md(trans_all)}

Transition analysis is important because it shows categorical movement, not just average movement.

## Layer Robustness

{table_md(fmt_layer(layer_df))}

Layer robustness separates peak performance from consistent performance across alphas.

## Alpha-Level Summary

{table_md(alpha_fmt)}

The alpha response is not assumed to be monotonic. A high alpha can help at one layer and hurt at another.

## Prompt-Level Steerability

Most positively shifted samples:

{table_md(prompt_top_fmt)}

Most negatively shifted samples:

{table_md(prompt_bottom_fmt)}

This shows whether steering works uniformly or only for a subset of prompts.

## Judge Agreement

{table_md(judge)}

Judge agreement explains why primary success and judge win ratio can differ. The pairwise judge may prefer steered output because of quality even when the primary score ties.

## Failure Keyword Mining

{table_md(fail_fmt)}

Failure keywords provide qualitative evidence for whether the main weakness is repetition, incoherence, irrelevance, or factual/logical error.

## Suggested Paper Claim

For {spec.title}, the safest claim is:

> Steering shows domain-specific layer-alpha behavior. The best configuration produces a measurable {spec.positive_shift_label}, but the practical value depends on quality preservation, clean success, transition behavior, and regression risk.

## Key Files

- `{spec.key}_config_summary.csv`
- `{spec.key}_sample_level_results.csv`
- `{spec.key}_layer_robustness.csv`
- `{spec.key}_alpha_summary.csv`
- `{spec.key}_prompt_steerability.csv`
- `{spec.key}_judge_agreement.csv`
- `{spec.key}_failure_keyword_counts.csv`
- `{spec.key}_heatmap_primary_delta.png`
- `{spec.key}_heatmap_quality_delta.png`
- `{spec.key}_heatmap_clean_directional_success_rate.png`
- `{spec.key}_transition_matrix_all_configs.png`
- `{spec.key}_pareto_primary_vs_quality.png`
"""
    (out_dir / f"{spec.title}_SUMMARY_REPORT.md").write_text(report, encoding="utf-8")


def generate_domain(spec: DomainSpec) -> None:
    out_dir = BASE_OUT / f"outputs_{spec.key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = parse_domain_files(spec)
    summary = summarize_configs(samples)
    layer_df = layer_robustness(summary)
    alpha_df = alpha_summary(summary)
    prompt_df = prompt_steerability(samples)
    judge_df = judge_agreement(samples)
    fail_df = failure_keywords(samples)

    samples.to_csv(out_dir / f"{spec.key}_sample_level_results.csv", index=False)
    summary.to_csv(out_dir / f"{spec.key}_config_summary.csv", index=False)
    layer_df.to_csv(out_dir / f"{spec.key}_layer_robustness.csv", index=False)
    alpha_df.to_csv(out_dir / f"{spec.key}_alpha_summary.csv", index=False)
    prompt_df.to_csv(out_dir / f"{spec.key}_prompt_steerability.csv", index=False)
    judge_df.to_csv(out_dir / f"{spec.key}_judge_agreement.csv")
    fail_df.to_csv(out_dir / f"{spec.key}_failure_keyword_counts.csv", index=False)
    transition_table(samples, spec.category_labels).to_csv(out_dir / f"{spec.key}_transition_matrix_all_configs.csv")

    heatmap(out_dir, summary, "primary_delta_mean", f"{spec.title}: Mean {spec.primary_name.title()} Shift", f"{spec.key}_heatmap_primary_delta", "RdBu", center=True, cbar_label="Mean steered - baseline primary")
    heatmap(out_dir, summary, "steered_primary_mean", f"{spec.title}: Absolute Steered Primary Score", f"{spec.key}_heatmap_steered_primary", "viridis", cbar_label="Mean steered primary")
    heatmap(out_dir, summary, "judge_steered_win_rate", f"{spec.title}: Judge Steered Win Rate", f"{spec.key}_heatmap_judge_steered_win_rate", "YlGn", fmt=".2f", cbar_label="Winner == Steered")
    heatmap(out_dir, summary, "primary_win_rate", f"{spec.title}: Primary Directional Success Rate", f"{spec.key}_heatmap_primary_win_rate", "YlGn", fmt=".2f", cbar_label="Share with primary delta > 0")
    heatmap(out_dir, summary, "clean_directional_success_rate", f"{spec.title}: Clean Directional Success Rate", f"{spec.key}_heatmap_clean_directional_success_rate", "YlGn", fmt=".2f", cbar_label="Directional shift plus quality thresholds")
    heatmap(out_dir, summary, "quality_delta_mean", f"{spec.title}: Mean Quality Delta", f"{spec.key}_heatmap_quality_delta", "RdBu", center=True, cbar_label="Mean quality delta")
    heatmap(out_dir, summary, "low_to_mid_or_high_rate", f"{spec.title}: Low-to-Mid-or-High Conversion", f"{spec.key}_heatmap_low_to_mid_or_high", "YlGn", fmt=".2f", cbar_label="P(steered >= 5 | baseline <= 3)")
    heatmap(out_dir, summary, "low_to_high_rate", f"{spec.title}: Low-to-High Conversion", f"{spec.key}_heatmap_low_to_high", "YlGn", fmt=".2f", cbar_label="P(steered >= 7 | baseline <= 3)")
    heatmap(out_dir, summary, "high_to_below_mid_rate", f"{spec.title}: High-to-Below-Mid Regression", f"{spec.key}_heatmap_high_to_below_mid", "OrRd", fmt=".2f", cbar_label="P(steered < 5 | baseline >= 7)")
    if spec.key == "politic":
        heatmap(out_dir, summary, "extremity_delta_mean", f"{spec.title}: Political Extremity Shift", f"{spec.key}_heatmap_extremity_delta", "RdBu_r", center=True, cbar_label="Mean abs(score-5) delta")
    alpha_response(out_dir, spec, summary)
    pareto(out_dir, spec, summary)
    win_tie_loss(out_dir, spec, summary)
    transition_heatmap(out_dir, spec, samples, f"{spec.title}: Category Transition Matrix", f"{spec.key}_transition_matrix_all_configs")

    write_report(out_dir, spec, samples, summary, layer_df, alpha_df, prompt_df, judge_df, fail_df)

    best = summary.loc[summary["primary_delta_mean"].idxmax()]
    positive_summary = summary[summary["primary_delta_mean"] > 0]
    quality_pool = positive_summary if not positive_summary.empty else summary
    clean_pool = positive_summary if not positive_summary.empty else summary
    best_quality = quality_pool.loc[quality_pool["quality_delta_mean"].idxmax()]
    best_clean = clean_pool.loc[clean_pool["clean_directional_success_rate"].idxmax()]
    print(
        f"{spec.title}: configs={len(summary)}, "
        f"best raw=L{int(best.layer)} a={best.alpha:g} delta={best.primary_delta_mean:+.3f}, "
        f"best quality=L{int(best_quality.layer)} a={best_quality.alpha:g} q={best_quality.quality_delta_mean:+.3f}, "
        f"best clean=L{int(best_clean.layer)} a={best_clean.alpha:g} clean={100 * best_clean.clean_directional_success_rate:.1f}%"
    )


def main() -> None:
    for key in ["logic", "politic", "sentiment"]:
        generate_domain(SPECS[key])


if __name__ == "__main__":
    main()
