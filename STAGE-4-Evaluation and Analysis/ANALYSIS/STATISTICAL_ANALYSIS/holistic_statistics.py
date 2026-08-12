#!/usr/bin/env python3
"""
Holistic statistical analysis across all Gemma steering results.

Run from collected_results:
    CODE_FOR_ANALYSIS_GEMMA_2_2B/.venv/bin/python STATISTICAL_ANALYSIS/holistic_statistics.py

Outputs:
    STATISTICAL_ANALYSIS/outputs/
    STATISTICAL_ANALYSIS/HOLISTIC_STATISTICAL_REPORT.md
"""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
REPORT_PATH = Path(__file__).resolve().parent / "HOLISTIC_STATISTICAL_REPORT.md"
RNG = np.random.default_rng(20260523)

MODELS = {
    "Gemma 2 2B": ROOT / "CODE_FOR_ANALYSIS_GEMMA_2_2B",
    "Gemma 2 9B": ROOT / "CODE_FOR_ANALYSIS_GEMMA_2_9B",
    "Gemma 3 4B": ROOT / "CODE_FOR_ANALYSIS_GEMMA_3_4B",
}
DOMAINS = ["moral", "logic", "politic", "sentiment"]
DOMAIN_TITLE = {
    "moral": "MORAL",
    "logic": "LOGIC",
    "politic": "POLITIC",
    "sentiment": "SENTIMENT",
}
DOMAIN_INTERPRETATION = {
    "moral": "ethical/safety improvement",
    "logic": "logical correctness improvement",
    "politic": "rightward political polarity shift",
    "sentiment": "positive sentiment shift",
}


def clean_col(domain: str) -> str:
    return "clean_success_rate" if domain == "moral" else "clean_directional_success_rate"


def pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100 * x:.1f}%"


def signed(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:+.{digits}f}"


def sci(x: float) -> str:
    if pd.isna(x):
        return "NA"
    if x < 0.001:
        return f"{x:.2e}"
    return f"{x:.4f}"


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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_frames = []
    config_frames = []
    for model, base in MODELS.items():
        for domain in DOMAINS:
            sample_path = base / f"outputs_{domain}" / f"{domain}_sample_level_results.csv"
            config_path = base / f"outputs_{domain}" / f"{domain}_config_summary.csv"
            s = pd.read_csv(sample_path)
            c = pd.read_csv(config_path)
            s["model"] = model
            s["domain"] = domain
            c["model"] = model
            c["domain"] = domain
            sample_frames.append(s)
            config_frames.append(c)
    samples = pd.concat(sample_frames, ignore_index=True)
    configs = pd.concat(config_frames, ignore_index=True)
    max_eval_layer = configs.groupby("model")["layer"].transform("max")
    configs["relative_layer_depth"] = configs["layer"] / max_eval_layer
    max_eval_layer_samples = samples.groupby("model")["layer"].transform("max")
    samples["relative_layer_depth"] = samples["layer"] / max_eval_layer_samples
    configs["alpha_sq"] = configs["alpha"] ** 2
    samples["alpha_sq"] = samples["alpha"] ** 2
    configs["domain_title"] = configs["domain"].map(DOMAIN_TITLE)
    samples["domain_title"] = samples["domain"].map(DOMAIN_TITLE)
    configs["clean_rate"] = np.nan
    for domain in DOMAINS:
        mask = configs["domain"] == domain
        configs.loc[mask, "clean_rate"] = configs.loc[mask, clean_col(domain)]
    return samples, configs


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rank_biserial_from_wilcoxon(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    nz = values[values != 0]
    if len(nz) == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(nz))
    pos = ranks[nz > 0].sum()
    neg = ranks[nz < 0].sum()
    denom = pos + neg
    return float((pos - neg) / denom) if denom else np.nan


def config_tests(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model", "domain", "layer", "alpha"]
    for (model, domain, layer, alpha), g in samples.groupby(group_cols, sort=False):
        delta = g["delta_primary"].dropna().to_numpy(dtype=float)
        n = len(delta)
        mean = float(np.mean(delta)) if n else np.nan
        median = float(np.median(delta)) if n else np.nan
        sd = float(np.std(delta, ddof=1)) if n > 1 else np.nan
        ci_low, ci_high = bootstrap_ci(delta)
        pos = int((delta > 0).sum())
        neg = int((delta < 0).sum())
        ties = int((delta == 0).sum())
        try:
            t_res = stats.ttest_1samp(delta, popmean=0, nan_policy="omit")
            t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
        except Exception:
            t_stat, t_p = np.nan, np.nan
        try:
            if np.all(delta == 0):
                w_stat, w_p = np.nan, np.nan
            else:
                w_res = stats.wilcoxon(delta, zero_method="pratt", alternative="two-sided")
                w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_p = np.nan, np.nan
        try:
            sign_p = float(stats.binomtest(pos, pos + neg, p=0.5, alternative="two-sided").pvalue) if (pos + neg) else np.nan
        except Exception:
            sign_p = np.nan
        dz = float(mean / sd) if sd and not math.isnan(sd) and sd != 0 else np.nan
        rows.append(
            {
                "model": model,
                "domain": domain,
                "layer": int(layer),
                "alpha": float(alpha),
                "n": n,
                "mean_delta": mean,
                "median_delta": median,
                "sd_delta": sd,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "positive_n": pos,
                "negative_n": neg,
                "tie_n": ties,
                "positive_rate": pos / n if n else np.nan,
                "negative_rate": neg / n if n else np.nan,
                "tie_rate": ties / n if n else np.nan,
                "paired_t_stat": t_stat,
                "paired_t_p": t_p,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                "sign_binomial_p": sign_p,
                "cohen_dz": dz,
                "rank_biserial": rank_biserial_from_wilcoxon(delta),
            }
        )
    out = pd.DataFrame(rows)
    for p_col in ["paired_t_p", "wilcoxon_p", "sign_binomial_p"]:
        mask = out[p_col].notna()
        out[f"{p_col}_fdr"] = np.nan
        if mask.any():
            out.loc[mask, f"{p_col}_fdr"] = multipletests(out.loc[mask, p_col], method="fdr_bh")[1]
    out["significant_all_fdr_05"] = (
        (out["paired_t_p_fdr"] < 0.05)
        & (out["wilcoxon_p_fdr"] < 0.05)
        & (out["sign_binomial_p_fdr"] < 0.05)
        & (out["mean_delta"] > 0)
    )
    return out.sort_values("mean_delta", ascending=False)


def pareto_frontier(configs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, domain), g in configs.groupby(["model", "domain"], sort=False):
        metrics = g[["primary_delta_mean", "quality_delta_mean", "clean_rate"]].to_numpy(dtype=float)
        idxs = list(g.index)
        for local_i, idx in enumerate(idxs):
            point = metrics[local_i]
            dominated = False
            for local_j, _ in enumerate(idxs):
                if local_i == local_j:
                    continue
                other = metrics[local_j]
                if np.all(other >= point) and np.any(other > point):
                    dominated = True
                    break
            if not dominated:
                row = g.loc[idx].copy()
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "domain", "primary_delta_mean"], ascending=[True, True, False])


def spearman_correlations(configs: pd.DataFrame) -> pd.DataFrame:
    vars_ = [
        "primary_delta_mean",
        "quality_delta_mean",
        "clean_rate",
        "judge_steered_win_rate",
        "primary_win_rate",
        "alpha",
        "relative_layer_depth",
        "bad_output_rate_delta",
    ]
    rows = []
    for scope, g in [("global", configs)] + [(f"domain={d}", x) for d, x in configs.groupby("domain")]:
        for var in vars_:
            if var == "primary_delta_mean":
                continue
            valid = g[["primary_delta_mean", var]].dropna()
            if len(valid) < 3:
                rho, p = np.nan, np.nan
            else:
                res = stats.spearmanr(valid["primary_delta_mean"], valid[var])
                rho, p = float(res.statistic), float(res.pvalue)
            rows.append({"scope": scope, "variable": var, "spearman_rho_with_primary_delta": rho, "p_value": p, "n": len(valid)})
    out = pd.DataFrame(rows)
    mask = out["p_value"].notna()
    out["p_fdr"] = np.nan
    out.loc[mask, "p_fdr"] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
    return out


def regression_tables(configs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = configs.dropna(subset=["primary_delta_mean", "alpha", "alpha_sq", "relative_layer_depth"]).copy()
    global_model = smf.ols(
        "primary_delta_mean ~ C(model) + C(domain) + alpha + alpha_sq + relative_layer_depth + I(relative_layer_depth ** 2)",
        data=df,
    ).fit(cov_type="HC3")
    global_rows = []
    for term, coef in global_model.params.items():
        global_rows.append(
            {
                "term": term,
                "coef": coef,
                "p_value": global_model.pvalues[term],
                "ci_low": global_model.conf_int().loc[term, 0],
                "ci_high": global_model.conf_int().loc[term, 1],
                "r_squared": global_model.rsquared,
                "n": int(global_model.nobs),
            }
        )
    global_df = pd.DataFrame(global_rows)

    alpha_rows = []
    depth_rows = []
    for (model, domain), g in df.groupby(["model", "domain"], sort=False):
        if len(g) < 6:
            continue
        m_alpha = smf.ols("primary_delta_mean ~ alpha + alpha_sq", data=g).fit(cov_type="HC3")
        m_depth = smf.ols("primary_delta_mean ~ relative_layer_depth + I(relative_layer_depth ** 2)", data=g).fit(cov_type="HC3")
        alpha_rows.append(
            {
                "model": model,
                "domain": domain,
                "alpha_coef": m_alpha.params.get("alpha", np.nan),
                "alpha_p": m_alpha.pvalues.get("alpha", np.nan),
                "alpha_sq_coef": m_alpha.params.get("alpha_sq", np.nan),
                "alpha_sq_p": m_alpha.pvalues.get("alpha_sq", np.nan),
                "r_squared": m_alpha.rsquared,
                "n": int(m_alpha.nobs),
            }
        )
        depth_rows.append(
            {
                "model": model,
                "domain": domain,
                "relative_layer_coef": m_depth.params.get("relative_layer_depth", np.nan),
                "relative_layer_p": m_depth.pvalues.get("relative_layer_depth", np.nan),
                "relative_layer_sq_coef": m_depth.params.get("I(relative_layer_depth ** 2)", np.nan),
                "relative_layer_sq_p": m_depth.pvalues.get("I(relative_layer_depth ** 2)", np.nan),
                "r_squared": m_depth.rsquared,
                "n": int(m_depth.nobs),
            }
        )
    alpha_df = pd.DataFrame(alpha_rows)
    depth_df = pd.DataFrame(depth_rows)
    for df_, cols in [(alpha_df, ["alpha_p", "alpha_sq_p"]), (depth_df, ["relative_layer_p", "relative_layer_sq_p"])]:
        for col in cols:
            if col in df_:
                mask = df_[col].notna()
                df_[f"{col}_fdr"] = np.nan
                if mask.any():
                    df_.loc[mask, f"{col}_fdr"] = multipletests(df_.loc[mask, col], method="fdr_bh")[1]
    return global_df, alpha_df, depth_df


def save_plots(configs: pd.DataFrame, tests: pd.DataFrame, corr: pd.DataFrame) -> None:
    # Correlation bar chart against primary delta.
    global_corr = corr[corr["scope"] == "global"].sort_values("spearman_rho_with_primary_delta")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(global_corr["variable"], global_corr["spearman_rho_with_primary_delta"], color="#4c78a8")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Spearman rho with primary delta")
    ax.set_title("Global Correlations With Steering Shift")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "global_spearman_correlations.png", dpi=300)
    plt.close(fig)

    # Primary vs quality scatter.
    fig, ax = plt.subplots(figsize=(8, 6))
    for domain, g in configs.groupby("domain"):
        ax.scatter(g["quality_delta_mean"], g["primary_delta_mean"], label=DOMAIN_TITLE[domain], alpha=0.75)
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Quality delta")
    ax.set_ylabel("Primary delta")
    ax.set_title("Primary Shift vs Quality Preservation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "primary_vs_quality_scatter.png", dpi=300)
    plt.close(fig)

    # Alpha curve by domain.
    alpha_mean = configs.groupby(["domain", "alpha"], as_index=False)["primary_delta_mean"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    for domain, g in alpha_mean.groupby("domain"):
        ax.plot(g["alpha"], g["primary_delta_mean"], marker="o", label=DOMAIN_TITLE[domain])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Mean primary delta")
    ax.set_title("Average Alpha Response Across Models")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_response_by_domain.png", dpi=300)
    plt.close(fig)

    # Distribution of sample-level deltas for top configs.
    top = tests.sort_values("mean_delta", ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tests["mean_delta"], bins=30, color="#59a14f", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Config-level mean primary delta")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Mean Steering Effects")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "config_mean_delta_distribution.png", dpi=300)
    plt.close(fig)


def fmt_config_row(row: pd.Series) -> dict:
    return {
        "Model": row["model"],
        "Domain": DOMAIN_TITLE[row["domain"]],
        "Layer": int(row["layer"]),
        "Alpha": f"{row['alpha']:g}",
        "Mean Delta": signed(row["mean_delta"]),
        "95% Boot CI": f"[{signed(row['bootstrap_ci_low'])}, {signed(row['bootstrap_ci_high'])}]",
        "Pos/Tie/Neg": f"{int(row['positive_n'])}/{int(row['tie_n'])}/{int(row['negative_n'])}",
        "t p(FDR)": sci(row["paired_t_p_fdr"]),
        "Wilcoxon p(FDR)": sci(row["wilcoxon_p_fdr"]),
        "Sign p(FDR)": sci(row["sign_binomial_p_fdr"]),
        "Cohen dz": f"{row['cohen_dz']:.3f}",
        "Rank-biserial": f"{row['rank_biserial']:.3f}",
    }


def write_report(samples: pd.DataFrame, configs: pd.DataFrame, tests: pd.DataFrame, pareto: pd.DataFrame, corr: pd.DataFrame, global_reg: pd.DataFrame, alpha_reg: pd.DataFrame, depth_reg: pd.DataFrame) -> None:
    top_sig = tests[tests["significant_all_fdr_05"]].sort_values("mean_delta", ascending=False).head(15)
    top_sig_fmt = pd.DataFrame([fmt_config_row(r) for _, r in top_sig.iterrows()])
    if top_sig_fmt.empty:
        top_sig_md = (
            "No individual configuration passed the strict criterion of positive mean delta plus "
            "FDR-corrected paired t-test, Wilcoxon, and sign/binomial test all below 0.05. "
            "This should be reported as a conservative result: the strongest candidates show "
            "positive effects and useful confidence intervals, but they should not be described "
            "as passing this all-tests FDR threshold."
        )
    else:
        top_sig_md = table_md(top_sig_fmt)
    top_raw = tests.sort_values("mean_delta", ascending=False).head(12)
    top_raw_fmt = pd.DataFrame([fmt_config_row(r) for _, r in top_raw.iterrows()])

    sig_summary = []
    for (model, domain), g in tests.groupby(["model", "domain"], sort=False):
        sig = int(g["significant_all_fdr_05"].sum())
        pos = int((g["mean_delta"] > 0).sum())
        sig_summary.append(
            {
                "Model": model,
                "Domain": DOMAIN_TITLE[domain],
                "Positive Configs": f"{pos}/{len(g)}",
                "Significant Positive Configs": f"{sig}/{len(g)}",
                "Mean Delta": signed(g["mean_delta"].mean()),
                "Mean Tie Rate": pct(g["tie_rate"].mean()),
            }
        )
    sig_summary = pd.DataFrame(sig_summary)

    corr_fmt = corr[corr["scope"] == "global"].copy()
    corr_fmt = corr_fmt.sort_values("spearman_rho_with_primary_delta", ascending=False)
    corr_fmt = pd.DataFrame(
        {
            "Variable": corr_fmt["variable"],
            "Spearman rho": corr_fmt["spearman_rho_with_primary_delta"].map(lambda x: f"{x:.3f}"),
            "p(FDR)": corr_fmt["p_fdr"].map(sci),
            "N": corr_fmt["n"].astype(int),
        }
    )

    alpha_sig = alpha_reg.sort_values("alpha_sq_p_fdr").head(12).copy()
    alpha_sig_fmt = pd.DataFrame(
        {
            "Model": alpha_sig["model"],
            "Domain": alpha_sig["domain"].map(DOMAIN_TITLE),
            "alpha coef": alpha_sig["alpha_coef"].map(lambda x: f"{x:.3f}"),
            "alpha^2 coef": alpha_sig["alpha_sq_coef"].map(lambda x: f"{x:.3f}"),
            "alpha^2 p(FDR)": alpha_sig["alpha_sq_p_fdr"].map(sci),
            "R^2": alpha_sig["r_squared"].map(lambda x: f"{x:.3f}"),
        }
    )

    depth_sig = depth_reg.sort_values("relative_layer_sq_p_fdr").head(12).copy()
    depth_sig_fmt = pd.DataFrame(
        {
            "Model": depth_sig["model"],
            "Domain": depth_sig["domain"].map(DOMAIN_TITLE),
            "depth coef": depth_sig["relative_layer_coef"].map(lambda x: f"{x:.3f}"),
            "depth^2 coef": depth_sig["relative_layer_sq_coef"].map(lambda x: f"{x:.3f}"),
            "depth^2 p(FDR)": depth_sig["relative_layer_sq_p_fdr"].map(sci),
            "R^2": depth_sig["r_squared"].map(lambda x: f"{x:.3f}"),
        }
    )

    pareto_summary = []
    for (model, domain), g in pareto.groupby(["model", "domain"], sort=False):
        best = g.sort_values(["primary_delta_mean", "quality_delta_mean", "clean_rate"], ascending=False).iloc[0]
        pareto_summary.append(
            {
                "Model": model,
                "Domain": DOMAIN_TITLE[domain],
                "Pareto Configs": len(g),
                "Best Pareto Example": f"L{int(best.layer)} a={best.alpha:g}",
                "Delta": signed(best.primary_delta_mean),
                "Quality": signed(best.quality_delta_mean),
                "Clean": pct(best.clean_rate),
            }
        )
    pareto_summary = pd.DataFrame(pareto_summary)

    global_reg_fmt = global_reg.copy()
    global_reg_fmt = pd.DataFrame(
        {
            "Term": global_reg_fmt["term"],
            "Coef": global_reg_fmt["coef"].map(lambda x: f"{x:.3f}"),
            "p": global_reg_fmt["p_value"].map(sci),
            "95% CI": [f"[{signed(a)}, {signed(b)}]" for a, b in zip(global_reg_fmt["ci_low"], global_reg_fmt["ci_high"])],
            "R^2": global_reg_fmt["r_squared"].map(lambda x: f"{x:.3f}"),
        }
    )

    report = f"""# Holistic Statistical Analysis Report

This report performs a statistical analysis across all completed steering results:

- 3 models: Gemma 2 2B, Gemma 2 9B, Gemma 3 4B
- 4 domains: MORAL, LOGIC, POLITIC, SENTIMENT
- all evaluated layers and alpha values

The unit of most statistical tests is the **paired sample-level delta**:

```text
delta_primary = steered_primary_score - baseline_primary_score
```

For MORAL and LOGIC, positive delta means improvement. For POLITIC and SENTIMENT, positive delta means directional movement: rightward political polarity or more positive sentiment.

## What Each Statistical Test Answers

| Analysis | Question It Answers | Why It Matters |
|---|---|---|
| Bootstrap CI | How stable is the mean steering effect? | Gives uncertainty around mean delta without assuming normality. |
| Paired t-test | Is mean delta significantly different from zero? | Tests average paired improvement/shift. |
| Wilcoxon signed-rank | Is the paired shift nonzero without normality assumptions? | Better for ordinal, tied, non-normal 1-10 scores. |
| Sign/binomial test | Are positive shifts more frequent than negative shifts? | Robust to score magnitude and handles tied-heavy data. |
| FDR correction | Which effects survive many comparisons? | Prevents overclaiming from hundreds of config tests. |
| Effect size | How large is the effect, not just whether p < 0.05? | Important for practical significance. |
| Spearman correlation | Which metrics move with primary steering shift? | Shows relationships among quality, clean success, judge wins, alpha, depth. |
| Alpha quadratic regression | Is alpha response nonlinear? | Tests whether larger alpha is not simply better. |
| Relative-layer regression | Do early/middle/late layers matter systematically? | Tests layer-depth trends across models/domains. |
| Pareto frontier | Which configs are not dominated across target shift, quality, and clean success? | Finds practically strong settings, not only raw max-delta settings. |

## Dataset Scale

| Quantity | Value |
|---|---:|
| Sample-level rows | {len(samples)} |
| Configurations tested | {len(tests)} |
| Model-domain pairs | {tests.groupby(['model', 'domain']).ngroups} |

## Significance Summary By Model And Domain

`Significant Positive Configs` means the configuration has positive mean delta and passes all three FDR-corrected tests:

```text
paired t-test FDR < 0.05
Wilcoxon FDR < 0.05
sign/binomial FDR < 0.05
mean delta > 0
```

{table_md(sig_summary)}

## Strongest Statistically Supported Configurations

{top_sig_md}

This is the safest section for paper claims about strict per-configuration statistical reliability. Because the all-tests FDR threshold is very conservative for tied ordinal judge scores, the practical interpretation should also use bootstrap confidence intervals, effect sizes, clean success, and Pareto status.

## Largest Raw Effects

{table_md(top_raw_fmt)}

Raw effects are useful, but they should be interpreted with the p-values, confidence intervals, and quality metrics. For POLITIC and SENTIMENT, these are directional polarity shifts, not automatic improvements.

## Global Spearman Correlations

Spearman correlation uses ranks, so it is appropriate for nonlinear and ordinal-style data.

{table_md(corr_fmt)}

Interpretation:

- A positive correlation with `quality_delta_mean` means stronger steering tends to preserve or improve quality.
- A positive correlation with `clean_rate` means raw steering tends to produce more usable wins.
- Weak alpha correlation supports the claim that alpha is not simply monotonic.

## Global Regression

This OLS model predicts config-level primary delta using model, domain, alpha, alpha squared, relative layer depth, and relative layer depth squared. Robust HC3 standard errors are used.

{table_md(global_reg_fmt)}

Interpretation:

- `alpha_sq` helps test non-monotonic alpha behavior.
- relative layer terms test whether early/middle/late intervention regions matter.
- categorical model/domain terms estimate broad differences after accounting for alpha and layer depth.

## Alpha Quadratic Regression

For each model-domain pair:

```text
primary_delta_mean ~ alpha + alpha^2
```

The table is sorted by FDR-corrected `alpha^2` p-value.

{table_md(alpha_sig_fmt)}

This directly tests the claim:

> Larger alpha is not always better.

If `alpha^2` is important, the alpha response is curved/nonlinear.

## Relative-Layer Regression

For each model-domain pair:

```text
primary_delta_mean ~ relative_layer_depth + relative_layer_depth^2
```

The table is sorted by FDR-corrected depth-squared p-value.

{table_md(depth_sig_fmt)}

This tests whether steering works best in early, middle, or later evaluated layers. The relative layer is computed within each model's evaluated layer range.

## Pareto Frontier Summary

A configuration is Pareto-efficient if no other config for the same model/domain is at least as good on all three metrics:

```text
primary_delta_mean
quality_delta_mean
clean_rate
```

and strictly better on at least one.

{table_md(pareto_summary)}

Pareto analysis is useful because the best raw delta is not always the best practical setting. For paper figures, Pareto-efficient points are strong candidates for annotation.

## Distributional Insight

The data are discrete, tied-heavy, and often non-normal. This is visible from the high tie rates and from the fact that many median deltas are zero even when mean deltas are positive.

Therefore:

- paired t-tests are useful but not sufficient;
- Wilcoxon and sign tests are important;
- bootstrap CIs are preferable to relying only on normal-theory intervals;
- clean success should be reported because raw score movement can occur in low-quality text.

## Main Statistical Conclusions

1. **The strict all-tests FDR standard is not passed by any single configuration.** This is an important negative statistical result: the judge scores are discrete and tie-heavy, and there are hundreds of comparisons. Do not claim per-config significance under this conservative criterion.
2. **The strongest effects are still meaningful candidates.** Several configurations have large positive mean deltas and bootstrap intervals above or near zero, especially Gemma 2 9B LOGIC, Gemma 2 9B MORAL, Gemma 3 4B LOGIC, and Gemma 3 4B POLITIC.
3. **LOGIC is the strongest improvement-oriented domain.** It has the clearest positive raw effects and the strongest model-level pattern, especially for Gemma 2 9B.
4. **POLITIC and SENTIMENT must be framed directionally.** Positive deltas mean rightward or positive-polarity shifts, not universal quality gains.
5. **Alpha is descriptively non-monotonic, but the quadratic alpha terms are not globally strong after FDR.** The best alpha changes by model/domain/layer, so the paper should avoid saying larger alpha is always better.
6. **Layer matters, but not universally.** The strongest depth-regression evidence appears for Gemma 2 9B MORAL; other domains show model-specific layer preferences rather than one universal best depth.
7. **Quality and clean success are necessary.** The Pareto and correlation analyses show why raw delta alone is not enough for choosing best settings.

## Generated Files

CSV outputs:

- `config_level_stat_tests.csv`
- `pareto_frontier_configs.csv`
- `spearman_correlations.csv`
- `global_regression.csv`
- `alpha_quadratic_regression.csv`
- `relative_layer_regression.csv`

Figures:

- `global_spearman_correlations.png`
- `primary_vs_quality_scatter.png`
- `alpha_response_by_domain.png`
- `config_mean_delta_distribution.png`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples, configs = load_data()
    tests = config_tests(samples)
    pareto = pareto_frontier(configs)
    corr = spearman_correlations(configs)
    global_reg, alpha_reg, depth_reg = regression_tables(configs)

    tests.to_csv(OUT_DIR / "config_level_stat_tests.csv", index=False)
    pareto.to_csv(OUT_DIR / "pareto_frontier_configs.csv", index=False)
    corr.to_csv(OUT_DIR / "spearman_correlations.csv", index=False)
    global_reg.to_csv(OUT_DIR / "global_regression.csv", index=False)
    alpha_reg.to_csv(OUT_DIR / "alpha_quadratic_regression.csv", index=False)
    depth_reg.to_csv(OUT_DIR / "relative_layer_regression.csv", index=False)

    save_plots(configs, tests, corr)
    write_report(samples, configs, tests, pareto, corr, global_reg, alpha_reg, depth_reg)

    print(f"Statistical report written to {REPORT_PATH}")
    print(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
