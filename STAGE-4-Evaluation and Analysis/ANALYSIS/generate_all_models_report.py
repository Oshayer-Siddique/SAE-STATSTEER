#!/usr/bin/env python3
"""
Generate one paper-facing report across all evaluated Gemma models and domains.

Run from this directory:
    CODE_FOR_ANALYSIS_GEMMA_2_2B/.venv/bin/python generate_all_models_report.py

Output:
    ALL_MODELS_ALL_DOMAINS_REPORT.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ALL_MODELS_ALL_DOMAINS_REPORT.md"

MODELS = {
    "Gemma 2 2B": ROOT / "CODE_FOR_ANALYSIS_GEMMA_2_2B",
    "Gemma 2 9B": ROOT / "CODE_FOR_ANALYSIS_GEMMA_2_9B",
    "Gemma 3 4B": ROOT / "CODE_FOR_ANALYSIS_GEMMA_3_4B",
}

DOMAINS = ["moral", "logic", "politic", "sentiment"]
DOMAIN_TITLES = {
    "moral": "MORAL",
    "logic": "LOGIC",
    "politic": "POLITIC",
    "sentiment": "SENTIMENT",
}
DOMAIN_INTERPRETATION = {
    "moral": "higher = more ethical/safe, so positive delta is an improvement",
    "logic": "higher = more correct, so positive delta is an improvement",
    "politic": "higher = more right-leaning, so positive delta is directional control, not automatic improvement",
    "sentiment": "higher = more positive, so positive delta is directional control unless positivity is the target",
}


def clean_col(domain: str) -> str:
    return "clean_success_rate" if domain == "moral" else "clean_directional_success_rate"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def signed(x: float, digits: int = 3) -> str:
    return f"{x:+.{digits}f}"


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


def read_config(model_name: str, base: Path, domain: str) -> pd.DataFrame:
    path = base / f"outputs_{domain}" / f"{domain}_config_summary.csv"
    df = pd.read_csv(path)
    df["model"] = model_name
    df["domain"] = domain
    return df


def read_layer(model_name: str, base: Path, domain: str) -> pd.DataFrame:
    path = base / f"outputs_{domain}" / f"{domain}_layer_robustness.csv"
    df = pd.read_csv(path)
    df["model"] = model_name
    df["domain"] = domain
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = []
    layers = []
    for model, base in MODELS.items():
        for domain in DOMAINS:
            configs.append(read_config(model, base, domain))
            layers.append(read_layer(model, base, domain))
    return pd.concat(configs, ignore_index=True), pd.concat(layers, ignore_index=True)


def domain_model_summary(configs: pd.DataFrame, layers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, domain), cfg in configs.groupby(["model", "domain"], sort=False):
        ccol = clean_col(domain)
        pos = cfg[cfg.primary_delta_mean > 0]
        pool = pos if len(pos) else cfg
        best_raw = cfg.loc[cfg.primary_delta_mean.idxmax()]
        best_quality = pool.loc[pool.quality_delta_mean.idxmax()]
        best_clean = pool.loc[pool[ccol].idxmax()]
        layer_df = layers[(layers.model == model) & (layers.domain == domain)]
        robust = layer_df.loc[layer_df.positive_config_rate.idxmax()]
        rows.append(
            {
                "Model": model,
                "Domain": DOMAIN_TITLES[domain],
                "Positive Configs": f"{len(pos)}/{len(cfg)}",
                "Mean Delta": signed(cfg.primary_delta_mean.mean()),
                "Best Raw": f"L{int(best_raw.layer)} a={best_raw.alpha:g} ({signed(best_raw.primary_delta_mean)})",
                "Best Quality-Positive": f"L{int(best_quality.layer)} a={best_quality.alpha:g} ({signed(best_quality.primary_delta_mean)}, q={signed(best_quality.quality_delta_mean)})",
                "Best Clean-Positive": f"L{int(best_clean.layer)} a={best_clean.alpha:g} ({pct(best_clean[ccol])})",
                "Most Robust Layer": f"L{int(robust.layer)} ({pct(robust.positive_config_rate)} positive)",
            }
        )
    return pd.DataFrame(rows)


def best_by_domain(configs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, cfg in configs.groupby("domain", sort=False):
        ccol = clean_col(domain)
        pos = cfg[cfg.primary_delta_mean > 0]
        pool = pos if len(pos) else cfg
        raw = cfg.loc[cfg.primary_delta_mean.idxmax()]
        quality = pool.loc[pool.quality_delta_mean.idxmax()]
        clean = pool.loc[pool[ccol].idxmax()]
        rows.append(
            {
                "Domain": DOMAIN_TITLES[domain],
                "Best Raw Across Models": f"{raw.model}, L{int(raw.layer)} a={raw.alpha:g}, {signed(raw.primary_delta_mean)}",
                "Best Quality-Positive Across Models": f"{quality.model}, L{int(quality.layer)} a={quality.alpha:g}, delta {signed(quality.primary_delta_mean)}, q {signed(quality.quality_delta_mean)}",
                "Best Clean-Positive Across Models": f"{clean.model}, L{int(clean.layer)} a={clean.alpha:g}, {pct(clean[ccol])}",
            }
        )
    return pd.DataFrame(rows)


def model_strength_table(configs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, cfg in configs.groupby("model", sort=False):
        pos = (cfg.primary_delta_mean > 0).sum()
        total = len(cfg)
        rows.append(
            {
                "Model": model,
                "Total Configs": total,
                "Positive Configs": f"{pos}/{total}",
                "Positive Rate": pct(pos / total),
                "Mean Delta Across All Domains": signed(cfg.primary_delta_mean.mean()),
                "Mean Quality Delta": signed(cfg.quality_delta_mean.mean()),
            }
        )
    return pd.DataFrame(rows)


def domain_strength_table(configs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, cfg in configs.groupby("domain", sort=False):
        pos = (cfg.primary_delta_mean > 0).sum()
        total = len(cfg)
        rows.append(
            {
                "Domain": DOMAIN_TITLES[domain],
                "Positive Configs": f"{pos}/{total}",
                "Positive Rate": pct(pos / total),
                "Mean Delta Across Models": signed(cfg.primary_delta_mean.mean()),
                "Best Model By Mean Delta": cfg.groupby("model").primary_delta_mean.mean().idxmax(),
            }
        )
    return pd.DataFrame(rows)


def raw_top_table(configs: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    top = configs.sort_values("primary_delta_mean", ascending=False).head(n)
    rows = []
    for _, r in top.iterrows():
        rows.append(
            {
                "Model": r.model,
                "Domain": DOMAIN_TITLES[r.domain],
                "Layer": int(r.layer),
                "Alpha": f"{r.alpha:g}",
                "Delta": signed(r.primary_delta_mean),
                "Quality Delta": signed(r.quality_delta_mean),
                "Primary Success": pct(r.primary_win_rate),
                "Judge Win": pct(r.judge_steered_win_rate),
                "Clean Success": pct(r[clean_col(r.domain)]),
            }
        )
    return pd.DataFrame(rows)


def write_report() -> None:
    configs, layers = load_all()
    model_domain = domain_model_summary(configs, layers)
    best_domain = best_by_domain(configs)
    model_strength = model_strength_table(configs)
    domain_strength = domain_strength_table(configs)
    top_raw = raw_top_table(configs)

    report = f"""# All Models, All Domains Steering Report

This report combines the completed analyses for three models:

- Gemma 2 2B
- Gemma 2 9B
- Gemma 3 4B

and four domains:

- MORAL
- LOGIC
- POLITIC
- SENTIMENT

It is designed as the high-level paper-facing synthesis across all completed experiments.

## Critical Interpretation Rule

The primary score does not mean the same thing in every domain.

| Domain | Interpretation |
|---|---|
| MORAL | {DOMAIN_INTERPRETATION['moral']} |
| LOGIC | {DOMAIN_INTERPRETATION['logic']} |
| POLITIC | {DOMAIN_INTERPRETATION['politic']} |
| SENTIMENT | {DOMAIN_INTERPRETATION['sentiment']} |

Therefore, use two different kinds of language:

- **Improvement-oriented:** MORAL and LOGIC.
- **Directional-control:** POLITIC and SENTIMENT.

This distinction is essential. A higher political score means more right-leaning, not more correct or better. A higher sentiment score means more positive, not necessarily more truthful or higher quality.

## Executive Summary Across All Models

{table_md(model_domain)}

## Best Results Across Models By Domain

{table_md(best_domain)}

## Model-Level Strength

{table_md(model_strength)}

Interpretation:

- Gemma 2 9B has the strongest overall steering profile by positive-config rate and mean delta.
- Gemma 3 4B is also strong, especially for LOGIC, POLITIC, and SENTIMENT selected regimes.
- Gemma 2 2B shows meaningful steering effects, but they are generally smaller and less robust.

## Domain-Level Strength

{table_md(domain_strength)}

Interpretation:

- LOGIC is the strongest improvement-oriented domain across models.
- MORAL has meaningful but more quality-constrained improvements.
- POLITIC and SENTIMENT require directional framing; their positive deltas indicate rightward/positive shifts, not universal improvement.

## Top Raw Shifts Across All Experiments

{table_md(top_raw)}

These are the largest raw primary-score shifts across all model-domain-layer-alpha configurations. They should not be interpreted equally across domains:

- LOGIC raw shifts are correctness improvements.
- MORAL raw shifts are ethical-alignment improvements.
- POLITIC raw shifts are rightward political shifts.
- SENTIMENT raw shifts are positive sentiment shifts.

## Core Cross-Model Insights

### 1. Larger Models Show Stronger Steering In LOGIC

LOGIC improves most clearly in the larger or stronger models:

- Gemma 2 9B: best raw LOGIC shift is `+1.160`.
- Gemma 3 4B: best raw LOGIC shift is `+0.750`.
- Gemma 2 2B: best raw LOGIC shift is `+0.590`.

This supports the interpretation that logical/correctness steering benefits from richer internal representations. The steering vector may be amplifying latent reasoning/correctness features that are more available in stronger models.

Paper-facing claim:

> LOGIC shows the clearest improvement-oriented steering response, especially in Gemma 2 9B and Gemma 3 4B, suggesting that correctness-oriented activation steering benefits from stronger model capacity.

### 2. MORAL Improves, But Quality Constraints Matter

Best raw MORAL shifts:

- Gemma 2 2B: `+0.550`
- Gemma 2 9B: `+0.700`
- Gemma 3 4B: `+0.380`

MORAL steering exists across models, but clean success remains much lower than raw shift. This means that ethical-alignment scores can improve while generations remain repetitive, low-richness, or low-coherence.

Paper-facing claim:

> MORAL steering produces measurable ethical-alignment shifts, but the best raw moral shift is not always the best practical setting once quality preservation and clean success are considered.

### 3. POLITIC Is Strongly Directional, Not Quality-Based

Best POLITIC shifts:

- Gemma 2 2B: `+0.610`
- Gemma 2 9B: `+0.612`
- Gemma 3 4B: `+0.796`

These are rightward political-polarity shifts. They are not "better" outputs unless the experimental goal is explicitly rightward steering.

Paper-facing claim:

> POLITIC steering demonstrates directional control over political polarity, with the strongest rightward shift appearing in Gemma 3 4B.

### 4. SENTIMENT Steering Is Strongest In 9B And 3 4B

Best SENTIMENT positive shifts:

- Gemma 2 2B: `+0.400`
- Gemma 2 9B: `+0.640`
- Gemma 3 4B: `+0.540`

This suggests positive-sentiment steering becomes stronger in the larger/newer models, although it still needs directional framing.

Paper-facing claim:

> SENTIMENT steering shows positive-polarity control, particularly in Gemma 2 9B and Gemma 3 4B, but should not be framed as general quality improvement unless positivity is the explicit target.

### 5. Best Layer Is Domain- And Model-Specific

There is no universal best layer.

Examples:

- Gemma 2 2B LOGIC: best raw layer `19`
- Gemma 2 9B LOGIC: best raw layer `19`
- Gemma 3 4B LOGIC: best raw layer `17`
- Gemma 2 9B MORAL: best raw layer `38`
- Gemma 3 4B MORAL: best raw layer `9`

This is a major paper insight:

> Activation steering is not layer-universal. The best intervention point varies by model scale, architecture/version, and target behavior.

### 6. Alpha Is Non-Monotonic

Across models and domains, stronger alpha does not reliably mean stronger useful steering.

Examples:

- Gemma 2 9B LOGIC: best raw alpha is `0.1`.
- Gemma 3 4B POLITIC: best raw alpha is `2.0`.
- Gemma 3 4B MORAL: best raw alpha is `0.7`.
- Gemma 2 2B SENTIMENT: best raw alpha is `0.2`.

Paper-facing claim:

> Steering strength has nonlinear effects; optimal alpha depends on both the model and the target domain.

## Recommended Main Paper Figure Set

For the main paper, use a compact set:

1. **All-model best-setting table**
   - one row per model-domain pair
   - best raw, best practical, clean success

2. **Primary-delta heatmap grid**
   - 3 models x 4 domains
   - use appendix if too large

3. **Layer robustness comparison**
   - shows positive-config rate by layer/model/domain

4. **Clean success comparison**
   - highlights the gap between raw shift and usable improvement

5. **Transition matrices for MORAL and LOGIC**
   - most interpretable for improvement-oriented domains

## Recommended Main Paper Claims

Strong claims supported by the analysis:

1. **Domain-specific steering:** Different behaviors require different layers and alphas.
2. **Model-specific steering:** The best intervention layer changes across Gemma 2 2B, Gemma 2 9B, and Gemma 3 4B.
3. **LOGIC is strongest:** Logical correctness shows the clearest improvement-oriented steering response.
4. **MORAL needs quality controls:** Ethical-alignment gains exist, but clean success and quality preservation are necessary.
5. **POLITIC/SENTIMENT are directional:** These domains show polarity control, not automatic improvement.
6. **Alpha is nonlinear:** Larger alpha is not reliably better.

## Claims To Avoid

Avoid:

- "Steering improves every model and every domain."
- "The same layer works across all models."
- "Higher alpha produces stronger useful steering."
- "Political steering is better when the score is higher."
- "Sentiment steering is better when the score is higher."
- "Raw primary delta alone proves useful generation improvement."

Use:

- "Steering produces model- and domain-specific directional shifts."
- "LOGIC and MORAL are improvement-oriented; POLITIC and SENTIMENT are directional-control domains."
- "Quality-preserving and clean-success metrics are necessary for interpreting steering."
- "Layer-alpha interaction is central to steering behavior."

## One-Paragraph Paper Summary

Across Gemma 2 2B, Gemma 2 9B, and Gemma 3 4B, activation steering produces measurable but highly model- and domain-specific behavioral shifts. LOGIC shows the strongest improvement-oriented response, especially in Gemma 2 9B, suggesting that correctness-oriented steering benefits from stronger latent reasoning representations. MORAL also improves in selected regimes, but clean-success analysis shows that ethical-alignment gains often require careful quality filtering. POLITIC and SENTIMENT demonstrate directional control over polarity dimensions rather than general quality improvements. Across all domains, the best layer and alpha vary substantially by model, showing that activation steering is not layer-universal and that steering strength has nonlinear effects.

## Links To Model-Level Reports

- [Gemma 2 2B combined report](CODE_FOR_ANALYSIS_GEMMA_2_2B/COMBINED_DOMAIN_REPORT.md)
- [Gemma 2 9B combined report](CODE_FOR_ANALYSIS_GEMMA_2_9B/COMBINED_DOMAIN_REPORT.md)
- [Gemma 3 4B combined report](CODE_FOR_ANALYSIS_GEMMA_3_4B/COMBINED_DOMAIN_REPORT.md)
"""
    OUT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    write_report()
