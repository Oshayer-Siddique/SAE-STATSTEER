#!/usr/bin/env python3
"""
Generate a combined report across MORAL, LOGIC, POLITIC, and SENTIMENT.

Run:
    CODE_FOR_ANALYSIS/.venv/bin/python CODE_FOR_ANALYSIS/generate_combined_report.py

Output:
    CODE_FOR_ANALYSIS/COMBINED_DOMAIN_REPORT.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "COMBINED_DOMAIN_REPORT.md"


DOMAINS = {
    "moral": {
        "title": "MORAL",
        "out": ROOT / "outputs_moral",
        "primary": "ethical alignment",
        "positive": "more ethical/safe",
        "interpretation": "Higher is better for the intended safety direction.",
        "clean_col": "clean_success_rate",
        "report": "MORAL_SUMMARY_REPORT.md",
        "heatmap": "moral_heatmap_primary_delta.png",
        "quality": "moral_heatmap_quality_delta.png",
        "pareto": "moral_pareto_primary_vs_quality.png",
    },
    "logic": {
        "title": "LOGIC",
        "out": ROOT / "outputs_logic",
        "primary": "logical correctness",
        "positive": "more correct / closer to ground truth",
        "interpretation": "Higher is better because the score is grounded in the target answer.",
        "clean_col": "clean_directional_success_rate",
        "report": "LOGIC_SUMMARY_REPORT.md",
        "heatmap": "logic_heatmap_primary_delta.png",
        "quality": "logic_heatmap_quality_delta.png",
        "pareto": "logic_pareto_primary_vs_quality.png",
    },
    "politic": {
        "title": "POLITIC",
        "out": ROOT / "outputs_politic",
        "primary": "political polarity",
        "positive": "more right-leaning/conservative",
        "interpretation": "Higher is not automatically better; it is a directional rightward polarity shift.",
        "clean_col": "clean_directional_success_rate",
        "report": "POLITIC_SUMMARY_REPORT.md",
        "heatmap": "politic_heatmap_primary_delta.png",
        "quality": "politic_heatmap_quality_delta.png",
        "pareto": "politic_pareto_primary_vs_quality.png",
    },
    "sentiment": {
        "title": "SENTIMENT",
        "out": ROOT / "outputs_sentiment",
        "primary": "sentiment polarity",
        "positive": "more positive sentiment",
        "interpretation": "Higher is not automatically better unless the target is positive sentiment.",
        "clean_col": "clean_directional_success_rate",
        "report": "SENTIMENT_SUMMARY_REPORT.md",
        "heatmap": "sentiment_heatmap_primary_delta.png",
        "quality": "sentiment_heatmap_quality_delta.png",
        "pareto": "sentiment_pareto_primary_vs_quality.png",
    },
}


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


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


def read_domain(key: str, info: dict) -> dict:
    out = info["out"]
    cfg = pd.read_csv(out / f"{key}_config_summary.csv")
    layer = pd.read_csv(out / f"{key}_layer_robustness.csv")
    alpha = pd.read_csv(out / f"{key}_alpha_summary.csv")
    fail = pd.read_csv(out / f"{key}_failure_keyword_counts.csv")
    prompt = pd.read_csv(out / f"{key}_prompt_steerability.csv")

    clean_col = info["clean_col"]
    positive = cfg[cfg["primary_delta_mean"] > 0]
    pool = positive if len(positive) else cfg

    best_raw = cfg.loc[cfg["primary_delta_mean"].idxmax()]
    best_quality = pool.loc[pool["quality_delta_mean"].idxmax()]
    best_clean = pool.loc[pool[clean_col].idxmax()]
    worst = cfg.loc[cfg["primary_delta_mean"].idxmin()]
    robust = layer.loc[layer["positive_config_rate"].idxmax()]
    volatile = layer.loc[layer["sd_primary_delta_across_alphas"].idxmax()]

    return {
        "cfg": cfg,
        "layer": layer,
        "alpha": alpha,
        "fail": fail,
        "prompt": prompt,
        "clean_col": clean_col,
        "positive": positive,
        "best_raw": best_raw,
        "best_quality": best_quality,
        "best_clean": best_clean,
        "worst": worst,
        "robust": robust,
        "volatile": volatile,
    }


def summarize_domains(data: dict) -> pd.DataFrame:
    rows = []
    for key, d in data.items():
        info = DOMAINS[key]
        cfg = d["cfg"]
        clean = d["clean_col"]
        br = d["best_raw"]
        bq = d["best_quality"]
        bc = d["best_clean"]
        robust = d["robust"]
        rows.append(
            {
                "Domain": info["title"],
                "Primary Meaning": info["positive"],
                "Configs": len(cfg),
                "Positive Configs": f"{len(d['positive'])}/{len(cfg)}",
                "Mean Delta": signed(cfg["primary_delta_mean"].mean()),
                "Best Raw": f"L{int(br.layer)} a={br.alpha:g} ({signed(br.primary_delta_mean)})",
                "Best Quality-Positive": f"L{int(bq.layer)} a={bq.alpha:g} ({signed(bq.primary_delta_mean)}, q={signed(bq.quality_delta_mean)})",
                "Best Clean-Positive": f"L{int(bc.layer)} a={bc.alpha:g} ({pct(bc[clean])})",
                "Most Robust Layer": f"L{int(robust.layer)} ({pct(robust.positive_config_rate)} positive)",
            }
        )
    return pd.DataFrame(rows)


def top_configs_table(data: dict) -> pd.DataFrame:
    rows = []
    for key, d in data.items():
        info = DOMAINS[key]
        cfg = d["cfg"].sort_values("primary_delta_mean", ascending=False).head(3)
        clean = d["clean_col"]
        for rank, (_, row) in enumerate(cfg.iterrows(), 1):
            rows.append(
                {
                    "Domain": info["title"],
                    "Rank": rank,
                    "Layer": int(row.layer),
                    "Alpha": f"{row.alpha:g}",
                    "Delta": signed(row.primary_delta_mean),
                    "Steered Avg": f"{row.steered_primary_mean:.2f}",
                    "Baseline Avg": f"{row.baseline_primary_mean:.2f}",
                    "Primary Success": pct(row.primary_win_rate),
                    "Judge Win": pct(row.judge_steered_win_rate),
                    "Clean Success": pct(row[clean]),
                    "Quality Delta": signed(row.quality_delta_mean),
                }
            )
    return pd.DataFrame(rows)


def layer_table(data: dict) -> pd.DataFrame:
    rows = []
    for key, d in data.items():
        info = DOMAINS[key]
        for _, row in d["layer"].iterrows():
            clean_col = "mean_clean_success_rate" if key == "moral" else "mean_clean_directional_success"
            rows.append(
                {
                    "Domain": info["title"],
                    "Layer": int(row.layer),
                    "Positive Configs": f"{int(row.positive_configs)}/{int(row.configs)}",
                    "Mean Delta": signed(row.mean_primary_delta),
                    "Best Alpha": f"{row.best_alpha:g}",
                    "Best Delta": signed(row.best_primary_delta),
                    "Volatility SD": f"{row.sd_primary_delta_across_alphas:.3f}",
                    "Mean Quality": signed(row.mean_quality_delta),
                    "Mean Clean": pct(row[clean_col]),
                }
            )
    return pd.DataFrame(rows)


def failure_table(data: dict) -> pd.DataFrame:
    rows = []
    for key, d in data.items():
        info = DOMAINS[key]
        top = d["fail"].head(5)
        rows.append(
            {
                "Domain": info["title"],
                "Top Failure Signals": ", ".join(
                    f"{r.keyword} ({pct(r.rate)})" for r in top.itertuples()
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report() -> None:
    data = {key: read_domain(key, info) for key, info in DOMAINS.items()}
    summary = summarize_domains(data)
    top = top_configs_table(data)
    layers = layer_table(data)
    failures = failure_table(data)

    report = f"""# Gemma 2 9B Combined Steering Evaluation Report

This report combines the completed analyses for:

- MORAL
- LOGIC
- POLITIC
- SENTIMENT

It is intended as a paper-facing synthesis: not every per-domain table is repeated, but the core cross-domain findings, useful caveats, and recommended claims are collected in one place.

## Critical Interpretation Rule

The primary score has different meanings across domains.

| Domain | Primary Score Meaning | How To Interpret Positive Delta |
|---|---|---|
| MORAL | Ethical/safety alignment | Higher is more ethical/safe, so positive delta is an improvement. |
| LOGIC | Correctness against ground truth | Higher is more correct, so positive delta is an improvement. |
| POLITIC | Political polarity, 1 left to 10 right | Positive delta means more right-leaning, not automatically better. |
| SENTIMENT | Sentiment polarity, 1 negative to 10 positive | Positive delta means more positive, not automatically better unless positivity is the target. |

This distinction is essential for paper writing. MORAL and LOGIC can be described as improvement-oriented domains. POLITIC and SENTIMENT should be described as **directional control** domains.

## Executive Summary

{table_md(summary)}

## Main Cross-Domain Findings

### 1. LOGIC Is The Strongest Improvement-Oriented Domain

LOGIC has the clearest broad improvement pattern:

```text
Positive configs: {len(data['logic']['positive'])}/{len(data['logic']['cfg'])}
Best raw setting: L{int(data['logic']['best_raw'].layer)} alpha {data['logic']['best_raw'].alpha:g}, delta {signed(data['logic']['best_raw'].primary_delta_mean)}
Most robust layer: L{int(data['logic']['robust'].layer)}
```

Layer `{int(data['logic']['robust'].layer)}` is especially strong for LOGIC. It is positive in most alpha settings and layer `{int(data['logic']['best_raw'].layer)}` has the best raw correctness shift. This suggests that the effective LOGIC intervention region for Gemma 2 9B is concentrated around these evaluated layers rather than uniformly distributed across the model.

Paper-facing interpretation:

> LOGIC steering shows the strongest evidence of broad improvement, with layer {int(data['logic']['best_raw'].layer)} producing the best raw correctness shift and layer {int(data['logic']['robust'].layer)} showing the most robust positive behavior across steering strengths.

### 2. MORAL Has A Strong Peak But Also Quality Constraints

MORAL has a meaningful best raw result:

```text
Best raw setting: L{int(data['moral']['best_raw'].layer)} alpha {data['moral']['best_raw'].alpha:g}, delta {signed(data['moral']['best_raw'].primary_delta_mean)}
Best practical setting: L{int(data['moral']['best_quality'].layer)} alpha {data['moral']['best_quality'].alpha:g}
Most robust layer: L{int(data['moral']['robust'].layer)}
```

The key MORAL finding is the separation between peak and practical performance:

- Layer {int(data['moral']['best_raw'].layer)} alpha {data['moral']['best_raw'].alpha:g} gives the strongest ethical-alignment shift.
- Layer {int(data['moral']['best_quality'].layer)} alpha {data['moral']['best_quality'].alpha:g} gives the best practical balance of positive moral shift and quality.
- Clean success remains low, so many moral gains are not fully usable generations.

Paper-facing interpretation:

> MORAL steering works in selected regimes, but the strongest raw alignment shift is not the same as the best practical setting once output quality is considered.

### 3. POLITIC Shows A Concentrated Rightward Shift

POLITIC should not be written as "better" or "worse." The primary score measures political polarity:

```text
1 = left/progressive
5 = center
10 = right/conservative
```

The result is directional:

```text
Best rightward shift: L{int(data['politic']['best_raw'].layer)} alpha {data['politic']['best_raw'].alpha:g}, delta {signed(data['politic']['best_raw'].primary_delta_mean)}
Positive configs: {len(data['politic']['positive'])}/{len(data['politic']['cfg'])}
Best clean-positive setting: L{int(data['politic']['best_clean'].layer)} alpha {data['politic']['best_clean'].alpha:g}
```

The strongest rightward effect appears at layer `{int(data['politic']['best_raw'].layer)}`, while the most robust rightward layer is `{int(data['politic']['robust'].layer)}`. This means POLITIC steering is localized, but for Gemma 2 9B the relevant layers differ from the smaller-model setting.

Paper-facing interpretation:

> POLITIC steering exhibits localized directional control: layer {int(data['politic']['best_raw'].layer)} produces the clearest rightward polarity shift, while robustness is concentrated around layer {int(data['politic']['robust'].layer)}.

### 4. SENTIMENT Shows A Narrow Positive-Sentiment Effect

SENTIMENT also needs directional framing:

```text
Positive delta = more positive sentiment
```

The strongest result is:

```text
L{int(data['sentiment']['best_raw'].layer)} alpha {data['sentiment']['best_raw'].alpha:g}, delta {signed(data['sentiment']['best_raw'].primary_delta_mean)}
```

{len(data['sentiment']['positive'])}/{len(data['sentiment']['cfg'])} configurations are positive. The best result is concentrated at layer {int(data['sentiment']['best_raw'].layer)} alpha {data['sentiment']['best_raw'].alpha:g}, while the most robust layer is {int(data['sentiment']['robust'].layer)}.

Paper-facing interpretation:

> SENTIMENT steering produces a positive shift in selected regimes, especially layer {int(data['sentiment']['best_raw'].layer)} alpha {data['sentiment']['best_raw'].alpha:g}, with broader robustness concentrated at layer {int(data['sentiment']['robust'].layer)}.

## Top Configurations By Domain

{table_md(top)}

## Layer Robustness Across Domains

{table_md(layers)}

## Cross-Domain Pattern

The most useful cross-domain insight is:

```text
Different domains prefer different layers.
```

Observed best raw settings:

- MORAL: layer {int(data['moral']['best_raw'].layer)}, alpha {data['moral']['best_raw'].alpha:g}
- LOGIC: layer {int(data['logic']['best_raw'].layer)}, alpha {data['logic']['best_raw'].alpha:g}
- POLITIC: layer {int(data['politic']['best_raw'].layer)}, alpha {data['politic']['best_raw'].alpha:g}
- SENTIMENT: layer {int(data['sentiment']['best_raw'].layer)}, alpha {data['sentiment']['best_raw'].alpha:g}

This suggests that steering behavior is not universal across attributes. Instead, layer choice depends on the target domain:

- LOGIC's strongest raw shift is at layer {int(data['logic']['best_raw'].layer)}.
- MORAL's strongest raw shift is at layer {int(data['moral']['best_raw'].layer)}.
- POLITIC's strongest rightward shift is at layer {int(data['politic']['best_raw'].layer)}.
- SENTIMENT's strongest positive shift is at layer {int(data['sentiment']['best_raw'].layer)}.

This is a strong paper insight:

> Activation steering is domain-localized: the most effective intervention layer varies by behavioral attribute.

## Alpha Behavior

Across domains, alpha response is not monotonic. Larger alpha does not always produce stronger or better steering.

Examples:

- MORAL: alpha {data['moral']['best_raw'].alpha:g} is best at layer {int(data['moral']['best_raw'].layer)}, while alpha {data['moral']['best_quality'].alpha:g} at layer {int(data['moral']['best_quality'].layer)} is better for quality-positive behavior.
- LOGIC: layer {int(data['logic']['best_raw'].layer)} alpha {data['logic']['best_raw'].alpha:g} is best raw, while layer {int(data['logic']['best_quality'].layer)} alpha {data['logic']['best_quality'].alpha:g} is best quality-positive.
- POLITIC: layer {int(data['politic']['best_raw'].layer)} alpha {data['politic']['best_raw'].alpha:g} gives the strongest rightward shift, while layer {int(data['politic']['best_quality'].layer)} alpha {data['politic']['best_quality'].alpha:g} gives the best quality-positive setting.
- SENTIMENT: layer {int(data['sentiment']['best_raw'].layer)} alpha {data['sentiment']['best_raw'].alpha:g} gives the strongest positive shift.

Paper-facing claim:

> Steering strength interacts nonlinearly with layer and domain; increasing alpha is not a reliable way to monotonically increase useful steering.

## Quality And Clean Success

Clean success is stricter than raw directional success:

```text
delta_primary > 0
AND steered_relevance >= 7
AND steered_richness >= 4
AND steered_coherence >= 4
```

Best clean-positive settings:

- MORAL: L{int(data['moral']['best_clean'].layer)} alpha {data['moral']['best_clean'].alpha:g}, {pct(data['moral']['best_clean'][data['moral']['clean_col']])}
- LOGIC: L{int(data['logic']['best_clean'].layer)} alpha {data['logic']['best_clean'].alpha:g}, {pct(data['logic']['best_clean'][data['logic']['clean_col']])}
- POLITIC: L{int(data['politic']['best_clean'].layer)} alpha {data['politic']['best_clean'].alpha:g}, {pct(data['politic']['best_clean'][data['politic']['clean_col']])}
- SENTIMENT: L{int(data['sentiment']['best_clean'].layer)} alpha {data['sentiment']['best_clean'].alpha:g}, {pct(data['sentiment']['best_clean'][data['sentiment']['clean_col']])}

Clean success is generally much lower than raw directional shift. This is important because it shows that steering can move the target attribute without always producing high-quality usable text.

Paper-facing claim:

> Clean-success analysis reveals a gap between attribute movement and usable generation quality, making quality-preserving evaluation necessary for activation steering.

## Failure Signals

{table_md(failures)}

The repeated appearance of repetition, coherence, and incoherence signals means that the dominant failure mode across domains is often generation quality, not only failure to shift the target attribute.

This helps justify why the paper should include both target-shift metrics and quality-preservation metrics.

## Recommended Main Paper Figures

Use a small number of high-value figures in the main paper:

1. **Four-domain primary delta heatmap grid**
   - MORAL, LOGIC, POLITIC, SENTIMENT
   - Shows domain-specific layer-alpha behavior.

2. **Pareto plot: target shift vs quality delta**
   - Either one combined figure or one per domain in appendix.
   - Shows best practical settings.

3. **Transition matrices**
   - Main text: MORAL and LOGIC.
   - Appendix: POLITIC and SENTIMENT.

4. **Layer robustness table**
   - Compact table showing best layer, positive config rate, and volatility.

## Recommended Main Paper Table

A compact table should include:

| Domain | Best Raw | Best Practical | Robust Layer | Main Interpretation |
|---|---|---|---|---|
| MORAL | L{int(data['moral']['best_raw'].layer)} a={data['moral']['best_raw'].alpha:g} | L{int(data['moral']['best_quality'].layer)} a={data['moral']['best_quality'].alpha:g} | L{int(data['moral']['robust'].layer)} | Ethical shift exists, but quality matters. |
| LOGIC | L{int(data['logic']['best_raw'].layer)} a={data['logic']['best_raw'].alpha:g} | L{int(data['logic']['best_quality'].layer)} a={data['logic']['best_quality'].alpha:g} | L{int(data['logic']['robust'].layer)} | Strongest improvement-oriented domain. |
| POLITIC | L{int(data['politic']['best_raw'].layer)} a={data['politic']['best_raw'].alpha:g} | L{int(data['politic']['best_quality'].layer)} a={data['politic']['best_quality'].alpha:g} | L{int(data['politic']['robust'].layer)} | Localized rightward polarity control. |
| SENTIMENT | L{int(data['sentiment']['best_raw'].layer)} a={data['sentiment']['best_raw'].alpha:g} | L{int(data['sentiment']['best_quality'].layer)} a={data['sentiment']['best_quality'].alpha:g} | L{int(data['sentiment']['robust'].layer)} | Positive sentiment shift in selected regimes. |

## Strongest Overall Paper Claim

The strongest combined claim is:

> Activation steering produces measurable but domain-specific behavioral shifts. LOGIC and MORAL show improvement-oriented effects, while POLITIC and SENTIMENT demonstrate directional control over polarity dimensions. The optimal layer and alpha vary substantially by domain, and the largest raw shift is not always the best practical setting once output quality, clean success, and regression behavior are considered.

## Claims To Avoid

Avoid these claims:

- "Steering improves all domains."
- "Higher alpha always improves steering."
- "Political steering is better when the score is higher."
- "Sentiment steering is better when the score is higher" unless the target is explicitly positive sentiment.
- "Judge win ratio alone proves success."

Use these instead:

- "Steering induces domain-specific directional shifts."
- "Some domains show improvement-oriented gains."
- "Layer-alpha interaction is central."
- "Quality-preserving and clean-success metrics are necessary."
- "POLITIC and SENTIMENT require directional interpretation."

## Report Links

- [MORAL report](outputs_moral/MORAL_SUMMARY_REPORT.md)
- [LOGIC report](outputs_logic/LOGIC_SUMMARY_REPORT.md)
- [POLITIC report](outputs_politic/POLITIC_SUMMARY_REPORT.md)
- [SENTIMENT report](outputs_sentiment/SENTIMENT_SUMMARY_REPORT.md)
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    write_report()
