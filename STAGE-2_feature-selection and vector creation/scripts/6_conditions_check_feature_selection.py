"""
unified_feature_selection.py

Unified feature selection pipeline using three-statistic consensus
(F-stat + MI + Cohen's d) for unidirectional steering.

Pipeline per (domain, layer):
1. Load F-stat, MI, and Cohen's d CSVs
2. Quality filter (5 conditions)
3. Target-direction restriction
4. Compute ranks per statistic (Borda count = mean rank)
5. Two-tier consensus (3/3 high, 2/3 medium)
6. Select top K features (K=16, 24, 32)
7. Save selections

Domain conventions:
    SENTIMENT: target = positive (d > 0 means positive). Layers: 13, 19, 23
    POLITIC:   target = right (d > 0 means right). Layers: 12, 16, 19, 23
    LOGIC:     target = correct (d > 0 means correct). Layers: 12, 16, 19, 23
        BUT csv uses correct/wrong columns and contrast = correct - wrong,
        so positive d = correct (target). Same convention.
    MORAL:     target = ethical (d > 0 means ethical). Layers: 12, 16, 19, 23
        BUT csv uses ethical/unethical columns and contrast = ethical - unethical,
        so positive d = ethical (target). Same convention.

So in ALL FOUR domains: positive d = target direction. Clean.
"""

import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")


# ============================================================
# DOMAIN CONFIGURATION
# ============================================================
DOMAIN_CONFIG = {
    "SENTIMENT": {
        "layers": [9, 17, 22, 29],
        "target_label": "positive",
        "anti_target_label": "negative",
        # Column names used in F-stat CSV for this domain
        "fstat_pos_col": "mean_positive_raw",
        "fstat_neg_col": "mean_negative_raw",
        "fstat_pos_rate_col": "activity_rate_positive",
        "fstat_neg_rate_col": "activity_rate_negative",
    },
    "POLITIC": {
        "layers": [9, 17, 22, 29],
        "target_label": "right",
        "anti_target_label": "left",
        "fstat_pos_col": "mean_right_raw",
        "fstat_neg_col": "mean_left_raw",
        "fstat_pos_rate_col": "activity_rate_right",
        "fstat_neg_rate_col": "activity_rate_left",
    },
    "LOGIC": {
        "layers": [9, 17, 22, 29],
        "target_label": "correct",
        "anti_target_label": "wrong",
        "fstat_pos_col": "mean_correct_raw",
        "fstat_neg_col": "mean_wrong_raw",
        "fstat_pos_rate_col": "activity_rate_correct",
        "fstat_neg_rate_col": "activity_rate_wrong",
    },
    "MORAL": {
        "layers": [9, 17, 22, 29],
        "target_label": "ethical",
        "anti_target_label": "unethical",
        "fstat_pos_col": "mean_ethical_raw",
        "fstat_neg_col": "mean_unethical_raw",
        "fstat_pos_rate_col": "activity_rate_ethical",
        "fstat_neg_rate_col": "activity_rate_unethical",
    },
}


# ============================================================
# LOADING
# ============================================================
def find_csv(directory, pattern_options):
    """Find CSV file matching one of the pattern options."""
    for pattern in pattern_options:
        candidates = list(Path(directory).glob(pattern))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(
        f"No CSV matching {pattern_options} in {directory}"
    )


def load_three_stat_files(domain, layer, base_dir):
    """
    Load F-stat, MI, and Cohen's d CSVs for a given (domain, layer).
    Returns three DataFrames sorted by feature_id with full SAE dimension.
    """
    domain_dir = Path(base_dir) / domain 

    fstat_dir = domain_dir / "fstat_complete" / f"layer_{layer}"
    mi_dir = domain_dir / "mi_complete" / f"layer_{layer}"
    d_dir = domain_dir / "cohens_d_complete" / f"layer_{layer}"

    fstat_path = find_csv(fstat_dir, [
        f"layer_{layer}_fstat_full_*.csv",
    ])
    mi_path = find_csv(mi_dir, [
        f"layer_{layer}_mi_full_*.csv",
    ])
    d_path = find_csv(d_dir, [
        f"layer_{layer}_d_full_*.csv",
    ])

    fstat = pd.read_csv(fstat_path)
    mi = pd.read_csv(mi_path)
    d = pd.read_csv(d_path)

    # Sort by feature_id for consistent indexing
    fstat = fstat.sort_values("feature_id").reset_index(drop=True)
    mi = mi.sort_values("feature_id").reset_index(drop=True)
    d = d.sort_values("feature_id").reset_index(drop=True)

    return {
        "fstat": fstat, "mi": mi, "d": d,
        "paths": {
            "fstat": str(fstat_path),
            "mi": str(mi_path),
            "d": str(d_path),
        },
    }


def merge_scores(loaded, domain_cfg):
    """
    Merge the three score files into a single DataFrame indexed by feature_id.
    Extracts only unique metrics from fstat and mi to prevent column suffix collisions
    with the metadata already present in Cohen's d.
    """
    # 1. Isolate only the specific metric columns we need from F-stat
    fstat_cols = ['feature_id', 'f_score', 'q_value_bh', 'eta_squared']
    fstat = loaded["fstat"][fstat_cols].copy()

    # 2. Isolate only the specific metric columns we need from MI
    mi_cols = ['feature_id', 'mi_score', 'mi_debiased',
               'nmi_score', 'q_value_perm_bh', 'mi_boot_cv', 'mi_zscore']
    mi = loaded["mi"][mi_cols].copy()

    # 3. Keep everything from Cohen's d (acts as the base for all shared metadata)
    d = loaded["d"].copy()

    # Drop any potential NaN feature_ids caused by trailing empty rows
    fstat = fstat.dropna(subset=["feature_id"])
    mi = mi.dropna(subset=["feature_id"])
    d = d.dropna(subset=["feature_id"])

    # Inner join safely aligns all rows. Since we stripped overlapping columns,
    # no _x or _y suffixes will be added!
    merged_df = d.merge(fstat, on="feature_id", how="inner")
    merged_df = merged_df.merge(mi, on="feature_id", how="inner")

    if len(merged_df) == 0:
        raise ValueError(
            "No common feature_ids found across the three score files.")

    merged = pd.DataFrame({
        "feature_id": merged_df["feature_id"].values,
        # F-stat
        "f_score": merged_df["f_score"].values,
        "f_qvalue": merged_df["q_value_bh"].values,
        "f_eta_squared": merged_df["eta_squared"].values,
        # MI
        "mi_score": merged_df["mi_score"].values,
        "mi_debiased": merged_df["mi_debiased"].values,
        "nmi_score": merged_df["nmi_score"].values,
        "mi_qvalue": merged_df["q_value_perm_bh"].values,
        "mi_boot_cv": merged_df["mi_boot_cv"].values,
        "mi_zscore": merged_df["mi_zscore"].values,
        # Cohen's d
        "cohens_d": merged_df["cohens_d_raw"].values,
        "abs_d": merged_df["abs_cohens_d_raw"].values,
        "d_robust": merged_df["d_robust_raw"].values,
        "hedges_g": merged_df["hedges_g_raw"].values,
        "d_qvalue": merged_df["q_value_raw_bh"].values,
        "d_ci_lower_analytical": merged_df["d_ci_lower_analytical"].values,
        "d_ci_upper_analytical": merged_df["d_ci_upper_analytical"].values,
        "d_ci_lower_bootstrap": merged_df["d_boot_ci_lower"].values,
        "d_ci_upper_bootstrap": merged_df["d_boot_ci_upper"].values,
        "ci_excludes_zero_bootstrap": merged_df["ci_excludes_zero_bootstrap"].values,
        "ci_excludes_zero_analytical": merged_df["ci_excludes_zero_analytical"].values,
        "sign_agrees_robust": merged_df["sign_agrees_with_robust"].values,
        "effect_category": merged_df["effect_category"].values,
        "direction_label": merged_df["direction_raw"].values,
        # Activity (guaranteed to exist exactly as named in 'd')
        "is_active": merged_df["is_active_feature"].values,
        "activity_rate_all": merged_df["activity_rate_all"].values,
        "activity_rate_pos": merged_df["activity_rate_pos"].values,
        "activity_rate_neg": merged_df["activity_rate_neg"].values,
        # Magnitude info
        "mean_positive_raw": merged_df["mean_positive_raw"].values,
        "mean_negative_raw": merged_df["mean_negative_raw"].values,
        "std_positive_raw": merged_df["std_positive_raw"].values,
        "std_negative_raw": merged_df["std_negative_raw"].values,
        "contrast_raw": merged_df["contrast_pos_minus_neg_raw"].values,
        # Interpretation
        "overlap_coefficient": merged_df["overlap_coefficient"].values,
        "probability_of_superiority": merged_df["probability_of_superiority"].values,
    })

    return merged


# ============================================================
# QUALITY FILTER (UNIDIRECTIONAL)
# ============================================================
def apply_quality_filter(
    merged,
    mi_cv_threshold=1.0,
    min_abs_d=0.3,
    require_one_sided_ci=True,
    require_significance=True,
    use_bootstrap_ci=True,
):
    """
    Relaxed unidirectional quality filter.

    Required (cannot be disabled):
    1. Feature is active
    2. Cohen's d > 0 (target direction)

    Optional (can be relaxed):
    3. d_ci_lower > 0 (one-sided CI) — controls reliability of direction
    4. |Cohen's d| >= min_abs_d — practical significance
    5. MI bootstrap CV < threshold — MI estimate stability
    6. q < 0.05 in any test — at least one statistical test agrees

    Note: sign_agrees_with_robust is NOT a filter (saved for inspection only).
    Reason: robust d is zero whenever class median is zero, which happens
    routinely for sparse SAE features. This makes it an unreliable filter
    for SAE data.

    Note: freq_alignment is NOT a filter. Reason: it's strongly correlated
    with target_direction already, and adds little additional signal.
    """
    has_bootstrap_ci = (
        merged["d_ci_lower_bootstrap"].abs().max() > 1e-6
        or merged["d_ci_upper_bootstrap"].abs().max() > 1e-6
    )

    if use_bootstrap_ci and has_bootstrap_ci:
        ci_lower = merged["d_ci_lower_bootstrap"].values
        ci_kind = "bootstrap"
    else:
        ci_lower = merged["d_ci_lower_analytical"].values
        ci_kind = "analytical"

    cond_active = merged["is_active"].values.astype(bool)
    cond_target_dir = merged["cohens_d"].values > 0
    cond_one_sided_ci = (
        ci_lower > 0 if require_one_sided_ci
        else np.ones(len(merged), dtype=bool)
    )
    cond_practical = merged["abs_d"].values >= min_abs_d
    cond_mi_stable = merged["mi_boot_cv"].values < mi_cv_threshold
    cond_significant = (
        ((merged["f_qvalue"].values < 0.05)
         | (merged["mi_qvalue"].values < 0.05)
         | (merged["d_qvalue"].values < 0.05))
        if require_significance
        else np.ones(len(merged), dtype=bool)
    )

    quality_mask = (
        cond_active
        & cond_target_dir
        & cond_one_sided_ci
        & cond_practical
        & cond_mi_stable
        & cond_significant
    )

    diagnostics = {
        "total_features": len(merged),
        "active": int(cond_active.sum()),
        "target_direction": int((cond_active & cond_target_dir).sum()),
        "one_sided_ci": int((cond_active & cond_target_dir & cond_one_sided_ci).sum()),
        "practical_d": int((cond_active & cond_target_dir & cond_one_sided_ci & cond_practical).sum()),
        "mi_stable": int((cond_active & cond_target_dir & cond_one_sided_ci & cond_practical & cond_mi_stable).sum()),
        "final_passed": int(quality_mask.sum()),
        "ci_kind_used": ci_kind,
        # Inspection only — not used as filter
        "would_pass_sign_agree": int((quality_mask & merged["sign_agrees_robust"].values.astype(bool)).sum()),
        "would_pass_freq_align": int((quality_mask & (merged["activity_rate_pos"].values > merged["activity_rate_neg"].values)).sum()),
    }

    return quality_mask, diagnostics


# ============================================================
# CONSENSUS RANKING
# ============================================================
def compute_three_ranks(merged, quality_mask):
    """
    Compute ranks for F, MI, |d| among quality-passed features.
    Features failing quality get rank = N+1 (worst).
    """
    n = len(merged)
    f_rank = np.full(n, n + 1, dtype=np.float64)
    mi_rank = np.full(n, n + 1, dtype=np.float64)
    d_rank = np.full(n, n + 1, dtype=np.float64)

    qf_indices = np.where(quality_mask)[0]

    if len(qf_indices) == 0:
        return f_rank, mi_rank, d_rank

    # Higher score = better, so we negate for ranking (rank 1 = best)
    f_rank[qf_indices] = rankdata(
        -merged["f_score"].values[qf_indices], method="average")
    mi_rank[qf_indices] = rankdata(
        -merged["mi_debiased"].values[qf_indices], method="average")
    d_rank[qf_indices] = rankdata(-merged["abs_d"].values[qf_indices],
                                  method="average")

    return f_rank, mi_rank, d_rank


def two_tier_consensus(merged, quality_mask, f_rank, mi_rank, d_rank, top_k_per_stat=300):
    """
    Compute high-confidence (3/3) and medium-confidence (2/3) tiers.
    Returns sets of feature indices (positions in merged).
    """
    qf_indices = set(np.where(quality_mask)[0])

    # Top-K features per statistic (only among quality-passed)
    top_f_set = set(np.where((f_rank <= top_k_per_stat) & quality_mask)[0])
    top_mi_set = set(np.where((mi_rank <= top_k_per_stat) & quality_mask)[0])
    top_d_set = set(np.where((d_rank <= top_k_per_stat) & quality_mask)[0])

    # High-confidence: all three agree
    high_conf = top_f_set & top_mi_set & top_d_set

    # Medium-confidence: 2 of 3 agree
    medium_conf = set()
    candidates = top_f_set | top_mi_set | top_d_set
    for feat in candidates:
        votes = (
            int(feat in top_f_set)
            + int(feat in top_mi_set)
            + int(feat in top_d_set)
        )
        if votes >= 2 and feat not in high_conf:
            medium_conf.add(feat)

    return high_conf, medium_conf, top_f_set, top_mi_set, top_d_set


def borda_count_ordering(features_set, f_rank, mi_rank, d_rank):
    """
    Order features by mean rank (Borda count).
    Lower mean rank = better.
    """
    feats = list(features_set)
    if not feats:
        return [], np.array([])

    mean_ranks = np.array([
        (f_rank[f] + mi_rank[f] + d_rank[f]) / 3.0
        for f in feats
    ])

    sort_idx = np.argsort(mean_ranks)
    ordered = [feats[i] for i in sort_idx]
    ordered_means = mean_ranks[sort_idx]

    return ordered, ordered_means


def select_top_k(high_conf_ordered, medium_conf_ordered, K):
    """
    Take top K: high-confidence first, then medium if needed.
    Returns selected feature indices and tier labels.
    """
    selected = []
    tiers = []

    # Take from high-confidence first
    for feat in high_conf_ordered:
        if len(selected) >= K:
            break
        selected.append(feat)
        tiers.append("high")

    # Fill remainder from medium
    if len(selected) < K:
        for feat in medium_conf_ordered:
            if len(selected) >= K:
                break
            selected.append(feat)
            tiers.append("medium")

    return selected, tiers


# ============================================================
# CONSENSUS DIAGNOSTIC (Spearman correlations)
# ============================================================
def consensus_diagnostic(merged, quality_mask):
    """
    Compute Spearman rank correlations between the three statistics
    on quality-passed features. This justifies use of multiple statistics.
    """
    from scipy.stats import spearmanr

    qf = quality_mask
    if qf.sum() < 10:
        return {
            "rho_f_mi": None, "rho_f_d": None, "rho_mi_d": None,
            "interpretation": "insufficient_features"
        }

    rho_f_mi, _ = spearmanr(
        merged["f_score"].values[qf], merged["mi_debiased"].values[qf])
    rho_f_d, _ = spearmanr(
        merged["f_score"].values[qf], merged["abs_d"].values[qf])
    rho_mi_d, _ = spearmanr(
        merged["mi_debiased"].values[qf], merged["abs_d"].values[qf])

    min_rho = min(rho_f_mi, rho_f_d, rho_mi_d)
    max_rho = max(rho_f_mi, rho_f_d, rho_mi_d)

    if min_rho > 0.95:
        interpretation = "highly_redundant_consensus_adds_little"
    elif max_rho < 0.4:
        interpretation = "weak_agreement_check_data_quality"
    else:
        interpretation = "informative_consensus"

    return {
        "rho_f_mi": float(rho_f_mi),
        "rho_f_d": float(rho_f_d),
        "rho_mi_d": float(rho_mi_d),
        "min_rho": float(min_rho),
        "max_rho": float(max_rho),
        "interpretation": interpretation,
    }


# ============================================================
# MAIN PIPELINE PER (DOMAIN, LAYER)
# ============================================================
def process_domain_layer(
    domain,
    layer,
    base_dir,
    out_dir,
    K_values,
    mi_cv_threshold,
    min_abs_d,
    top_k_per_stat,
    use_bootstrap_ci,
    require_one_sided_ci,
    require_significance,    # <-- ADD THIS
):
    """Run the full pipeline for one (domain, layer)."""
    domain_cfg = DOMAIN_CONFIG[domain]

    print(f"\n{'='*70}")
    print(f"DOMAIN: {domain} | LAYER: {layer}")
    print(f"  Target direction: {domain_cfg['target_label']} (d > 0)")
    print(f"{'='*70}")

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------
    try:
        loaded = load_three_stat_files(domain, layer, base_dir)
    except FileNotFoundError as e:
        print(f"  SKIPPING — file not found: {e}")
        return None

    merged = merge_scores(loaded, domain_cfg)
    sae_dim = len(merged)

    # ------------------------------------------------------------
    # Quality filter
    # ------------------------------------------------------------
    quality_mask, diagnostics = apply_quality_filter(
        merged,
        mi_cv_threshold=mi_cv_threshold,
        min_abs_d=min_abs_d,
        require_one_sided_ci=require_one_sided_ci,
        require_significance=require_significance,
        use_bootstrap_ci=use_bootstrap_ci,
    )

    print(
        f"\n  Quality filter cascade (CI kind: {diagnostics['ci_kind_used']}):")
    print(f"    Total features:           {diagnostics['total_features']}")
    print(f"    + active:                 {diagnostics['active']}")
    print(f"    + target direction (d>0): {diagnostics['target_direction']}")
    print(f"    + one-sided CI > 0:       {diagnostics['one_sided_ci']}")
    print(f"    + |d| >= {min_abs_d}:           {diagnostics['practical_d']}")
    print(
        f"    + MI CV < {mi_cv_threshold}:           {diagnostics['mi_stable']}")
    print(f"    + q<0.05 (any test):      {diagnostics['final_passed']}")
    print(f"  Inspection (not filtered):")
    print(
        f"    Would also pass sign_agrees_robust: {diagnostics['would_pass_sign_agree']}/{diagnostics['final_passed']}")
    print(
        f"    Would also pass freq_alignment:     {diagnostics['would_pass_freq_align']}/{diagnostics['final_passed']}")
    # ------------------------------------------------------------
    # Consensus diagnostic
    # ------------------------------------------------------------
    consensus_diag = consensus_diagnostic(merged, quality_mask)
    print(f"\n  Spearman correlations (quality-passed features):")
    print(f"    F vs MI:  rho = {consensus_diag['rho_f_mi']}")
    print(f"    F vs |d|: rho = {consensus_diag['rho_f_d']}")
    print(f"    MI vs |d|:rho = {consensus_diag['rho_mi_d']}")
    print(f"    Verdict: {consensus_diag['interpretation']}")

    # ------------------------------------------------------------
    # Compute ranks
    # ------------------------------------------------------------
    f_rank, mi_rank, d_rank = compute_three_ranks(merged, quality_mask)

    # ------------------------------------------------------------
    # Two-tier consensus
    # ------------------------------------------------------------
    high_conf, medium_conf, top_f_set, top_mi_set, top_d_set = two_tier_consensus(
        merged, quality_mask, f_rank, mi_rank, d_rank, top_k_per_stat=top_k_per_stat,
    )

    print(f"\n  Consensus (top_k_per_stat={top_k_per_stat}):")
    print(f"    Top by F-stat:    {len(top_f_set)}")
    print(f"    Top by MI:        {len(top_mi_set)}")
    print(f"    Top by |d|:       {len(top_d_set)}")
    print(f"    High-conf (3/3):  {len(high_conf)}")
    print(f"    Medium-conf (2/3):{len(medium_conf)}")

    # ------------------------------------------------------------
    # Borda ordering within tiers
    # ------------------------------------------------------------
    high_ordered, high_means = borda_count_ordering(
        high_conf, f_rank, mi_rank, d_rank)
    medium_ordered, medium_means = borda_count_ordering(
        medium_conf, f_rank, mi_rank, d_rank)

    # ------------------------------------------------------------
    # Save selections for each K
    # ------------------------------------------------------------
    domain_out = Path(out_dir) / domain
    layer_out = domain_out / f"layer_{layer}"
    layer_out.mkdir(parents=True, exist_ok=True)

    selections_summary = {}

    for K in K_values:
        selected_indices, tiers = select_top_k(high_ordered, medium_ordered, K)

        if len(selected_indices) < K:
            print(
                f"\n  WARNING: K={K} requested but only {len(selected_indices)} consensus features available")
        
            if len(selected_indices) == 0:
                sel_df = pd.DataFrame(columns=[
                    "selection_rank", "tier", "feature_id", "cohens_d",
                    # ... (won't be used, but prevents crash)
                ])
                # Save empty CSV so downstream knows
                csv_path = layer_out / f"{domain}_layer{layer}_K{K}.csv"
                npy_path = layer_out / f"{domain}_layer{layer}_K{K}_feature_ids.npy"
                sel_df.to_csv(csv_path, index=False)
                np.save(npy_path, np.array([], dtype=np.int32))
                
                selections_summary[K] = {
                    "n_selected": 0,
                    "n_high_conf": 0,
                    "n_medium_conf": 0,
                    "mean_d": 0.0,
                    "min_d": 0.0,
                    "max_d": 0.0,
                    "n_all_three_significant": 0,
                    "csv_path": str(csv_path),
                    "npy_path": str(npy_path),
                }
                continue  # skip the rest of the loop body

        # Build selection DataFrame
        selection_data = []
        for rank, (idx, tier) in enumerate(zip(selected_indices, tiers), start=1):
            selection_data.append({
                "selection_rank": rank,
                "tier": tier,
                "feature_id": int(merged["feature_id"].iloc[idx]),
                "cohens_d": float(merged["cohens_d"].iloc[idx]),
                "abs_d": float(merged["abs_d"].iloc[idx]),
                "hedges_g": float(merged["hedges_g"].iloc[idx]),
                "d_robust": float(merged["d_robust"].iloc[idx]),
                "f_score": float(merged["f_score"].iloc[idx]),
                "mi_score": float(merged["mi_score"].iloc[idx]),
                "mi_debiased": float(merged["mi_debiased"].iloc[idx]),
                "nmi_score": float(merged["nmi_score"].iloc[idx]),
                "f_rank": float(f_rank[idx]),
                "mi_rank": float(mi_rank[idx]),
                "d_rank": float(d_rank[idx]),
                "mean_rank": float((f_rank[idx] + mi_rank[idx] + d_rank[idx]) / 3.0),
                "f_qvalue": float(merged["f_qvalue"].iloc[idx]),
                "mi_qvalue": float(merged["mi_qvalue"].iloc[idx]),
                "d_qvalue": float(merged["d_qvalue"].iloc[idx]),
                "d_ci_lower_analytical": float(merged["d_ci_lower_analytical"].iloc[idx]),
                "d_ci_upper_analytical": float(merged["d_ci_upper_analytical"].iloc[idx]),
                "activity_rate_all": float(merged["activity_rate_all"].iloc[idx]),
                "activity_rate_pos": float(merged["activity_rate_pos"].iloc[idx]),
                "activity_rate_neg": float(merged["activity_rate_neg"].iloc[idx]),
                "mean_positive_raw": float(merged["mean_positive_raw"].iloc[idx]),
                "mean_negative_raw": float(merged["mean_negative_raw"].iloc[idx]),
                "contrast_raw": float(merged["contrast_raw"].iloc[idx]),
                "effect_category": str(merged["effect_category"].iloc[idx]),
                "overlap_coefficient": float(merged["overlap_coefficient"].iloc[idx]),
                "probability_of_superiority": float(merged["probability_of_superiority"].iloc[idx]),
                "all_three_q_lt_0_05": bool(
                    (merged["f_qvalue"].iloc[idx] < 0.05)
                    and (merged["mi_qvalue"].iloc[idx] < 0.05)
                    and (merged["d_qvalue"].iloc[idx] < 0.05)
                ),
            })

        sel_df = pd.DataFrame(selection_data)

        # Save CSV and NPY
        csv_path = layer_out / f"{domain}_layer{layer}_K{K}.csv"
        npy_path = layer_out / f"{domain}_layer{layer}_K{K}_feature_ids.npy"

        sel_df.to_csv(csv_path, index=False)
        np.save(npy_path, sel_df["feature_id"].values.astype(np.int32))

        # K-level summary
        if len(sel_df) > 0:
            high_count = (sel_df["tier"] == "high").sum()
            med_count = (sel_df["tier"] == "medium").sum()
            mean_d = sel_df["cohens_d"].mean()
            min_d = sel_df["cohens_d"].min()
            max_d = sel_df["cohens_d"].max()
            full_consensus = sel_df["all_three_q_lt_0_05"].sum()
        else:
            high_count = med_count = full_consensus = 0
            mean_d = min_d = max_d = 0

        selections_summary[K] = {
            "n_selected": len(sel_df),
            "n_high_conf": int(high_count),
            "n_medium_conf": int(med_count),
            "mean_d": float(mean_d),
            "min_d": float(min_d),
            "max_d": float(max_d),
            "n_all_three_significant": int(full_consensus),
            "csv_path": str(csv_path),
            "npy_path": str(npy_path),
        }

        print(f"\n  K={K}: selected {len(sel_df)} features "
              f"({high_count} high, {med_count} medium)")
        print(f"    d range: [{min_d:.3f}, {max_d:.3f}], mean d: {mean_d:.3f}")
        print(
            f"    Features w/ q<0.05 in all 3 tests: {full_consensus}/{len(sel_df)}")

    # ------------------------------------------------------------
    # Save layer-level summary JSON
    # ------------------------------------------------------------
    summary = {
        "domain": domain,
        "layer": int(layer),
        "target_label": domain_cfg["target_label"],
        "anti_target_label": domain_cfg["anti_target_label"],
        "sae_dim": int(sae_dim),
        "input_paths": loaded["paths"],
        "filter_params": {
            "mi_cv_threshold": float(mi_cv_threshold),
            "min_abs_d": float(min_abs_d),
            "use_bootstrap_ci": bool(use_bootstrap_ci),
            "require_one_sided_ci": bool(require_one_sided_ci),
            "require_significance": bool(require_significance),
            "top_k_per_stat": int(top_k_per_stat),
        },
        "filter_cascade": diagnostics,
        "consensus_diagnostic": consensus_diag,
        "consensus_counts": {
            "high_confidence": len(high_conf),
            "medium_confidence": len(medium_conf),
            "total_consensus": len(high_conf) + len(medium_conf),
        },
        "selections": selections_summary,
    }

    summary_path = layer_out / f"{domain}_layer{layer}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved: {summary_path}")

    return summary


# ============================================================
# RUN ALL DOMAINS
# ============================================================
def run_all(args):
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    K_values = sorted(set(args.K_values))

    # Determine which domains to run
    if args.domains:
        domains_to_run = [d.upper() for d in args.domains]
    else:
        domains_to_run = list(DOMAIN_CONFIG.keys())

    all_summaries = []

    for domain in domains_to_run:
        if domain not in DOMAIN_CONFIG:
            print(f"\nUNKNOWN DOMAIN: {domain}, skipping")
            continue

        domain_cfg = DOMAIN_CONFIG[domain]

        # Determine layers
        if args.layers:
            layers_to_run = args.layers
        else:
            layers_to_run = domain_cfg["layers"]

        for layer in layers_to_run:
            if layer not in domain_cfg["layers"]:
                print(
                    f"\nLayer {layer} not in {domain} config (available: {domain_cfg['layers']}), skipping")
                continue

            # In run_all, replace the call with:
            summary = process_domain_layer(
                domain=domain,
                layer=layer,
                base_dir=base_dir,
                out_dir=out_dir,
                K_values=K_values,
                mi_cv_threshold=args.mi_cv_threshold,
                min_abs_d=args.min_abs_d,
                top_k_per_stat=args.top_k_per_stat,
                use_bootstrap_ci=not args.no_bootstrap_ci,
                require_one_sided_ci=not args.no_one_sided_ci,
                require_significance=not args.no_significance,
            )

            if summary:
                all_summaries.append(summary)

    # Master summary
    master_path = out_dir / "all_selections_summary.json"
    with open(master_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Master overview table
    overview_rows = []
    for s in all_summaries:
        for K, sel in s["selections"].items():
            overview_rows.append({
                "domain": s["domain"],
                "layer": s["layer"],
                "target": s["target_label"],
                "K": K,
                "n_selected": sel["n_selected"],
                "n_high_conf": sel["n_high_conf"],
                "n_medium_conf": sel["n_medium_conf"],
                "mean_d": sel["mean_d"],
                "min_d": sel["min_d"],
                "max_d": sel["max_d"],
                "n_all_three_significant": sel["n_all_three_significant"],
                "quality_passed": s["filter_cascade"]["final_passed"],
                "rho_f_mi": s["consensus_diagnostic"]["rho_f_mi"],
                "rho_f_d": s["consensus_diagnostic"]["rho_f_d"],
                "rho_mi_d": s["consensus_diagnostic"]["rho_mi_d"],
            })

    overview_df = pd.DataFrame(overview_rows)
    overview_csv = out_dir / "all_selections_overview.csv"
    overview_df.to_csv(overview_csv, index=False)

    print("\n" + "="*70)
    print("ALL DONE")
    print("="*70)
    print(f"Master summary: {master_path}")
    print(f"Overview table: {overview_csv}")
    print(f"\nProcessed {len(all_summaries)} (domain, layer) combinations")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified consensus feature selection for unidirectional steering."
    )
    parser.add_argument(
        "--base_dir",
        default=".",
        help="Base directory containing SENTIMENT/, POLITIC/, LOGIC/, MORAL/ folders.",
    )
    parser.add_argument(
        "--out_dir",
        default="./final_selections",
        help="Output directory for selected features.",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Domains to process. Default: all four.",
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        help="Layers to process. Default: all configured for each domain.",
    )
    parser.add_argument(
        "--K_values",
        nargs="+",
        type=int,
        default=[16, 24, 32],
        help="Subspace sizes to select.",
    )
    parser.add_argument(
        "--mi_cv_threshold",
        type=float,
        default=1.0,
        help="MI bootstrap CV threshold (lower = stricter). Default 1.0.",
    )
    parser.add_argument(
        "--min_abs_d",
        type=float,
        default=0.5,
        help="Minimum |Cohen's d| for practical significance. Default 0.5.",
    )
    parser.add_argument(
        "--top_k_per_stat",
        type=int,
        default=300,
        help="Top-K cutoff per statistic for consensus filtering. Default 300.",
    )
    parser.add_argument(
        "--no_bootstrap_ci",
        action="store_true",
        help="Use analytical CI instead of bootstrap (auto-fallback if bootstrap unavailable).",
    )
    parser.add_argument(
        "--no_one_sided_ci",
        action="store_true",
        help="Disable one-sided CI requirement.",
    )
    # Replace these two:
    # parser.add_argument("--no_freq_alignment", ...)
    # parser.add_argument("--no_sign_agree", ...)

    # With this single addition:
    parser.add_argument(
        "--no_significance",
        action="store_true",
        help="Disable q<0.05 requirement (most permissive — only use if very few features survive).",
    )

    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
