"""
cohens_d_analysis.py

Complete Cohen's d (effect size) analysis for SAE features.

Why Cohen's d matters for SAE feature selection:
    F-statistic tells you IF means differ (statistical significance).
    Cohen's d tells you HOW MUCH means differ (practical significance).
    
    A feature with F=500, d=0.1 is "significant but trivial"
    A feature with F=50,  d=2.0 is "significant AND large effect"
    
    For steering, magnitude matters: a feature must shift activation
    enough to actually influence output, not just differ measurably.
    
    Critically: Cohen's d is SIGNED. Unlike F-stat or MI (both unsigned),
    d tells you DIRECTION:
        d > 0 → feature fires more for positive class
        d < 0 → feature fires more for negative class
    This is what you'll use to initialize the SSV steering vector.

Mathematical foundation:
    Standard Cohen's d:
        d = (mean_pos - mean_neg) / s_pooled
        s_pooled = sqrt((s_pos^2 + s_neg^2) / 2)
    
    Hedges' g (small-sample correction):
        g = d * (1 - 3 / (4*N - 9))
        where N = n_pos + n_neg
    
    Glass's Δ (used when class variances differ greatly):
        Δ = (mean_pos - mean_neg) / s_neg
        (uses control/negative class SD only)
    
    Effect size interpretation (Cohen 1988):
        |d| < 0.2  trivial
        |d| ~ 0.5  medium
        |d| > 0.8  large
        |d| > 1.5  very large
        |d| > 3.0  massive (rare in noisy data, common for clean SAE features)

Outputs (per layer):
    - Full CSV: all effect size metrics per feature
    - Top-K CSV: ranked top features by |d|
    - NPZ: raw arrays for fast downstream loading
    - JSON summary
    - Diagnostic plots
"""

import os
import json
import argparse
import warnings
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "results", ".cache")
DEFAULT_MPLCONFIGDIR = os.path.join(PROJECT_ROOT, "results", ".matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", DEFAULT_CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", DEFAULT_MPLCONFIGDIR)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from joblib import Parallel, delayed

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_BASE_DIR = os.path.join(PROJECT_ROOT, "results", "sae_features_gemma3_4b")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "results", "cohens_d_complete")
DEFAULT_LAYERS = [12, 19, 26, 31, 38]
LABEL_1_NAME = "right"
LABEL_0_NAME = "left"


# ============================================================
# UTILS
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def as_str_array(x):
    return np.array([str(v).strip().lower() for v in x])


def safe_json_dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked_p = p[order]
    q = ranked_p * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    q_values = np.empty_like(q)
    q_values[order] = q
    return q_values


def apply_transform(Z, transform):
    if transform == "raw":
        return Z
    if transform == "log1p":
        return np.log1p(Z)
    raise ValueError(f"Unknown transform: {transform}")


def pool_if_needed(Z, pool_method="max"):
    if Z.ndim == 2:
        return Z
    if Z.ndim == 3:
        print(f"Detected 3D Z: {Z.shape}")
        if pool_method == "max":
            return Z.max(axis=1)
        if pool_method == "mean":
            return Z.mean(axis=1)
        raise ValueError(f"Unknown pool_method: {pool_method}")
    raise ValueError(f"Unsupported Z shape: {Z.shape}")


def validate_loaded_data(Z, y):
    if len(Z) != len(y):
        raise ValueError(f"Z rows and y length mismatch: Z={Z.shape}, y={y.shape}")
    unique_labels = sorted(set(y.tolist()))
    if unique_labels != [0, 1]:
        raise ValueError(f"Expected y labels [0, 1], got {unique_labels}")
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both classes for Cohen's d.")
    if Z.shape[0] < 4:
        raise ValueError("Too few rows for Cohen's d.")
    if Z.shape[1] < 2:
        raise ValueError("Too few SAE features.")


# ============================================================
# LOADING (consistent with F-stat and MI)
# ============================================================
def load_layer_file(base_dir, layer):
    path = os.path.join(base_dir, f"layer_{layer}_all_sae_features.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    data = np.load(path, allow_pickle=True)
    for key in ["Z", "y"]:
        if key not in data.files:
            raise KeyError(f"Missing key '{key}' in {path}. Found keys: {data.files}")
    Z = data["Z"]
    y = data["y"].astype(np.int32)
    Z = pool_if_needed(Z)
    Z = np.asarray(Z, dtype=np.float32)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    validate_loaded_data(Z, y)
    metadata = {}
    for key in ["pair_ids", "sides", "split2", "topics", "sources"]:
        if key in data.files:
            metadata[key] = data[key]
    return path, Z, y, metadata, data.files


def filter_rows(Z, y, metadata, allowed_splits=None):
    if allowed_splits is None or len(allowed_splits) == 0:
        row_mask = np.ones(len(y), dtype=bool)
        return Z, y, metadata, row_mask
    if "split2" not in metadata:
        raise ValueError("allowed_splits provided but split2 missing.")
    split2 = as_str_array(metadata["split2"])
    allowed = set([str(s).strip().lower() for s in allowed_splits])
    row_mask = np.array([s in allowed for s in split2], dtype=bool)
    if row_mask.sum() == 0:
        raise ValueError(f"No rows matched allowed_splits={allowed_splits}")
    Z_use = Z[row_mask]
    y_use = y[row_mask]
    metadata_use = {k: v[row_mask] if len(v) == len(row_mask) else v
                    for k, v in metadata.items()}
    validate_loaded_data(Z_use, y_use)
    return Z_use, y_use, metadata_use, row_mask


# ============================================================
# CORE EFFECT SIZE COMPUTATIONS
# ============================================================
def compute_pooled_std(s_pos, s_neg, n_pos, n_neg, weighted=True):
    """
    Pooled standard deviation.
    
    Two conventions exist:
    
    1. Sample-size-weighted (weighted=True), default in most software:
        s_pooled = sqrt( ((n_pos-1)*s_pos^2 + (n_neg-1)*s_neg^2) / (n_pos+n_neg-2) )
       This is what t-tests use. Best when class sizes are very different.
    
    2. Simple average (weighted=False), Cohen's original definition:
        s_pooled = sqrt( (s_pos^2 + s_neg^2) / 2 )
       Best when class sizes are similar.
    
    For balanced designs (your case, presumably), they're nearly identical.
    We compute both and let the user choose. We report the weighted version
    by default since it's more common in modern statistical literature.
    """
    if weighted:
        numerator = (n_pos - 1) * s_pos**2 + (n_neg - 1) * s_neg**2
        denominator = n_pos + n_neg - 2
        return np.sqrt(numerator / denominator + 1e-12)
    else:
        return np.sqrt((s_pos**2 + s_neg**2) / 2.0 + 1e-12)


def compute_cohens_d(mean_pos, mean_neg, s_pooled):
    """
    Cohen's d = (mean_pos - mean_neg) / s_pooled
    
    SIGNED:
        d > 0 → feature fires more for positive class
        d < 0 → feature fires more for negative class
        d ≈ 0 → no class difference
    
    Magnitude:
        |d| ≈ 0.2  small
        |d| ≈ 0.5  medium  
        |d| ≈ 0.8  large
    """
    return (mean_pos - mean_neg) / (s_pooled + 1e-12)


def compute_hedges_g(d, n_total):
    """
    Hedges' g: small-sample bias-corrected version of Cohen's d.
    
    Cohen's d is biased upward in small samples. The bias factor is:
        J(N) = 1 - 3 / (4*N - 9)
        
    where N = n_pos + n_neg.
    
    Hedges' g = d * J(N)
    
    For N > 50, J ≈ 1 and g ≈ d.
    For small samples (N < 30), the correction matters.
    
    Why this matters for SAE features:
        With ~50-100 samples per class, the bias correction is real.
        Reviewers familiar with effect size literature will appreciate
        seeing both d and g reported.
    """
    if n_total <= 9:
        # Correction undefined for very small N; return d unchanged with warning
        return d, np.nan
    correction = 1.0 - (3.0 / (4.0 * n_total - 9.0))
    return d * correction, correction


def compute_glass_delta(mean_pos, mean_neg, s_pos, s_neg):
    """
    Glass's Δ: uses ONLY one group's SD as the standardizer.
    
    Δ_neg-as-control = (mean_pos - mean_neg) / s_neg
    Δ_pos-as-control = (mean_pos - mean_neg) / s_pos
    
    Use case: when class variances differ MASSIVELY (heteroscedasticity).
    Pooling variances makes no sense if one class has σ=0.1 and the other
    σ=10. Glass's Δ uses one group's SD as a stable reference.
    
    For SAE features, this happens with binary-like features:
        - Negative class: feature is always 0 (s_neg ≈ 0)
        - Positive class: feature is sometimes large (s_pos > 0)
    Glass's Δ with s_pos as standardizer handles this gracefully.
    
    We compute both versions; the one with the LARGER denominator is
    more conservative (smaller |Δ|).
    """
    delta_neg_ref = (mean_pos - mean_neg) / (s_neg + 1e-12)
    delta_pos_ref = (mean_pos - mean_neg) / (s_pos + 1e-12)
    return delta_neg_ref, delta_pos_ref


def compute_d_confidence_interval(d, n_pos, n_neg, alpha=0.05):
    """
    95% confidence interval for Cohen's d.
    
    Uses the non-central t-distribution exactly, but the closed-form
    approximation (Hedges & Olkin 1985) is sufficient for our purposes:
    
        SE(d) = sqrt( (n_pos + n_neg) / (n_pos * n_neg) + d^2 / (2 * (n_pos + n_neg)) )
        CI = d ± z_{alpha/2} * SE(d)
    
    Why this matters:
        A feature with d = 0.8, CI = [0.7, 0.9] is reliably "large effect".
        A feature with d = 0.8, CI = [-0.2, 1.8] is uncertain — could be tiny or huge.
        
    Filter heuristic:
        Keep features where the CI does NOT include 0 (or some threshold).
        This is the effect-size analog of statistical significance.
    """
    n_total = n_pos + n_neg
    se = np.sqrt(
        n_total / (n_pos * n_neg + 1e-12)
        + d**2 / (2.0 * n_total + 1e-12)
    )
    z = sp_stats.norm.ppf(1 - alpha / 2)  # 1.96 for alpha=0.05
    ci_lower = d - z * se
    ci_upper = d + z * se
    return ci_lower, ci_upper, se


def compute_overlap_coefficient(d):
    """
    Overlap coefficient (OVL): proportion of overlap between the two
    class distributions, assuming both are Gaussian with same variance.
    
        OVL = 2 * Φ(-|d| / 2)
    
    where Φ is the standard normal CDF.
    
    Interpretation:
        OVL = 1.0  → distributions identical (d = 0)
        OVL = 0.5  → moderate separation (d ≈ 1.35)
        OVL = 0.1  → strong separation (d ≈ 3.3)
        OVL = 0.0  → no overlap (d → ∞)
    
    More intuitive than d for non-statisticians.
    Useful sanity check: features with OVL > 0.9 are essentially indistinguishable
    between classes regardless of what F-stat says.
    """
    return 2.0 * sp_stats.norm.cdf(-np.abs(d) / 2.0)


def compute_probability_of_superiority(d):
    """
    Probability of Superiority (Cohen's U3):
    P(X_pos > X_neg) for randomly drawn samples, assuming Gaussian.
    
        PoS = Φ(d / sqrt(2))
    
    Interpretation:
        PoS = 0.5  → no difference
        PoS = 0.7  → 70% of pos samples exceed a random neg sample
        PoS = 0.95 → very strong separation
    
    Highly intuitive for non-technical audiences and good for paper exposition.
    """
    return sp_stats.norm.cdf(d / np.sqrt(2.0))


def compute_robust_effect_size(Z_pos_feat, Z_neg_feat):
    """
    Robust alternative to Cohen's d using median and MAD.
    
        d_robust = (median_pos - median_neg) / MAD_pooled
    
    MAD = median absolute deviation = median(|x - median(x)|)
    MAD_pooled = sqrt((MAD_pos^2 + MAD_neg^2) / 2)
    
    Why:
        Cohen's d is sensitive to outliers because it uses mean and SD.
        SAE activations have extreme outliers (a feature might fire
        with magnitude 50 once and 0.1 normally). The robust version
        is unaffected by outliers.
    
    If d_robust >> d, outliers are inflating Cohen's d artificially.
    If d_robust << d, outliers are masking a real effect.
    """
    median_pos = np.median(Z_pos_feat)
    median_neg = np.median(Z_neg_feat)
    mad_pos = np.median(np.abs(Z_pos_feat - median_pos))
    mad_neg = np.median(np.abs(Z_neg_feat - median_neg))
    # MAD is biased estimator of σ for Gaussian; multiply by 1.4826 to get SD-equivalent
    sd_pos_robust = 1.4826 * mad_pos
    sd_neg_robust = 1.4826 * mad_neg
    pooled_robust = np.sqrt((sd_pos_robust**2 + sd_neg_robust**2) / 2.0 + 1e-12)
    return (median_pos - median_neg) / (pooled_robust + 1e-12)


def compute_robust_effect_sizes_vectorized(Z_pos, Z_neg):
    """Vectorized version of robust effect size for all features at once."""
    median_pos = np.median(Z_pos, axis=0)
    median_neg = np.median(Z_neg, axis=0)
    mad_pos = np.median(np.abs(Z_pos - median_pos[None, :]), axis=0)
    mad_neg = np.median(np.abs(Z_neg - median_neg[None, :]), axis=0)
    sd_pos_robust = 1.4826 * mad_pos
    sd_neg_robust = 1.4826 * mad_neg
    pooled_robust = np.sqrt((sd_pos_robust**2 + sd_neg_robust**2) / 2.0 + 1e-12)
    return (median_pos - median_neg) / (pooled_robust + 1e-12)


# ============================================================
# T-TEST FOR P-VALUES (Cohen's d's natural significance test)
# ============================================================
def welch_t_test_vectorized(Z_pos, Z_neg):
    """
    Welch's t-test for each feature. Doesn't assume equal variances.
    Cohen's d's natural significance test.
    
    Returns:
        t_stat, df, p_value (two-sided)
    """
    n_pos = Z_pos.shape[0]
    n_neg = Z_neg.shape[0]
    
    mean_pos = Z_pos.mean(axis=0)
    mean_neg = Z_neg.mean(axis=0)
    var_pos = Z_pos.var(axis=0, ddof=1)
    var_neg = Z_neg.var(axis=0, ddof=1)
    
    # Welch's t
    se = np.sqrt(var_pos / n_pos + var_neg / n_neg + 1e-12)
    t_stat = (mean_pos - mean_neg) / se
    
    # Welch-Satterthwaite degrees of freedom
    numerator = (var_pos / n_pos + var_neg / n_neg) ** 2
    denominator = (
        (var_pos / n_pos) ** 2 / (n_pos - 1 + 1e-12)
        + (var_neg / n_neg) ** 2 / (n_neg - 1 + 1e-12)
    )
    df = numerator / (denominator + 1e-12)
    
    # Two-sided p-value
    p_value = 2.0 * sp_stats.t.sf(np.abs(t_stat), df)
    p_value = np.nan_to_num(p_value, nan=1.0, posinf=1.0, neginf=1.0)
    
    return t_stat, df, p_value


# ============================================================
# BOOTSTRAP CI FOR COHEN'S D (more robust than analytical CI)
# ============================================================
def bootstrap_cohens_d(
    Z_pos, Z_neg,
    n_bootstrap=500,
    alpha=0.05,
    random_state=42,
    n_jobs=-1,
):
    """
    Bootstrap percentile CI for Cohen's d.
    
    Why bootstrap when we have analytical CI?
        The analytical CI assumes Gaussian distributions.
        SAE features are decidedly non-Gaussian.
        Bootstrap CI makes no distributional assumptions.
    
    For the consensus stage, we want features whose d is RELIABLE.
    A bootstrap CI that excludes 0 (or our threshold) is strong evidence.
    """
    rng = np.random.default_rng(random_state)
    n_pos = Z_pos.shape[0]
    n_neg = Z_neg.shape[0]
    n_features = Z_pos.shape[1]
    
    def single_bootstrap(seed):
        local_rng = np.random.default_rng(seed)
        idx_pos = local_rng.integers(0, n_pos, size=n_pos)
        idx_neg = local_rng.integers(0, n_neg, size=n_neg)
        Z_pos_b = Z_pos[idx_pos]
        Z_neg_b = Z_neg[idx_neg]
        mean_pos_b = Z_pos_b.mean(axis=0)
        mean_neg_b = Z_neg_b.mean(axis=0)
        s_pos_b = Z_pos_b.std(axis=0, ddof=1)
        s_neg_b = Z_neg_b.std(axis=0, ddof=1)
        s_pooled_b = np.sqrt(
            ((n_pos - 1) * s_pos_b**2 + (n_neg - 1) * s_neg_b**2)
            / (n_pos + n_neg - 2)
            + 1e-12
        )
        return (mean_pos_b - mean_neg_b) / (s_pooled_b + 1e-12)
    
    seeds = rng.integers(0, 10**9, size=n_bootstrap)
    
    print(f"  Running {n_bootstrap} bootstrap iterations for Cohen's d CI...")
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(single_bootstrap)(seed) for seed in seeds
    )
    elapsed = time.time() - t0
    print(f"  Bootstrap completed in {elapsed:.1f}s")
    
    boot_matrix = np.stack(results, axis=0)  # (n_bootstrap, n_features)
    
    return {
        "d_boot_mean": boot_matrix.mean(axis=0),
        "d_boot_std": boot_matrix.std(axis=0, ddof=1),
        "d_boot_ci_lower": np.percentile(boot_matrix, 100 * alpha / 2, axis=0),
        "d_boot_ci_upper": np.percentile(boot_matrix, 100 * (1 - alpha / 2), axis=0),
    }


# ============================================================
# MAIN COHEN'S D ANALYSIS
# ============================================================
def compute_complete_cohens_d(
    Z,
    y,
    layer,
    transform="log1p",
    min_activity_rate=0.01,
    min_active_count=5,
    do_bootstrap=True,
    n_bootstrap=500,
    do_robust=True,
    do_glass=True,
    n_jobs=-1,
    random_state=42,
):
    """
    Complete Cohen's d pipeline.
    
    Computes:
        1. Cohen's d on raw and transformed activations
        2. Hedges' g (small-sample correction)
        3. Glass's Δ (heteroscedasticity-robust)
        4. Robust d (median + MAD, outlier-robust)
        5. Welch's t-test p-values
        6. FDR correction
        7. Analytical and bootstrap CIs
        8. Overlap coefficient and probability of superiority (interpretive)
    """
    n_rows, sae_dim = Z.shape
    pos_mask = y == 1
    neg_mask = y == 0
    Z_pos = Z[pos_mask]
    Z_neg = Z[neg_mask]
    n_pos = Z_pos.shape[0]
    n_neg = Z_neg.shape[0]
    n_total = n_pos + n_neg
    
    print(f"  Data: {n_rows} rows, {sae_dim} features, {n_pos} pos, {n_neg} neg")
    
    # ------------------------------------------------------------
    # Raw descriptive statistics
    # ------------------------------------------------------------
    mean_all_raw = Z.mean(axis=0)
    mean_pos_raw = Z_pos.mean(axis=0)
    mean_neg_raw = Z_neg.mean(axis=0)
    s_pos_raw = Z_pos.std(axis=0, ddof=1)
    s_neg_raw = Z_neg.std(axis=0, ddof=1)
    var_all_raw = Z.var(axis=0, ddof=1)
    contrast_raw = mean_pos_raw - mean_neg_raw
    
    active_count_all = np.count_nonzero(Z > 0, axis=0)
    active_count_pos = np.count_nonzero(Z_pos > 0, axis=0)
    active_count_neg = np.count_nonzero(Z_neg > 0, axis=0)
    activity_rate_all = active_count_all / n_rows
    activity_rate_pos = active_count_pos / n_pos
    activity_rate_neg = active_count_neg / n_neg
    
    direction = np.where(
        contrast_raw > 0, LABEL_1_NAME,
        np.where(contrast_raw < 0, LABEL_0_NAME, "neutral"),
    )
    
    # ------------------------------------------------------------
    # Activity filter
    # ------------------------------------------------------------
    active_mask = (
        (activity_rate_all >= min_activity_rate)
        & (active_count_all >= min_active_count)
        & (var_all_raw > 1e-12)
    )
    active_feature_ids = np.where(active_mask)[0]
    n_active = len(active_feature_ids)
    
    if n_active == 0:
        raise ValueError("No active features after filtering.")
    
    print(f"  Active features: {n_active} / {sae_dim} ({100*n_active/sae_dim:.1f}%)")
    
    # ------------------------------------------------------------
    # Transformed activations (apply same transform as F-stat / MI)
    # ------------------------------------------------------------
    Z_stat = apply_transform(Z, transform)
    Z_pos_stat = Z_stat[pos_mask]
    Z_neg_stat = Z_stat[neg_mask]
    
    mean_pos_stat = Z_pos_stat.mean(axis=0)
    mean_neg_stat = Z_neg_stat.mean(axis=0)
    s_pos_stat = Z_pos_stat.std(axis=0, ddof=1)
    s_neg_stat = Z_neg_stat.std(axis=0, ddof=1)
    
    # ------------------------------------------------------------
    # Cohen's d (both raw and transformed)
    # ------------------------------------------------------------
    print("  Computing Cohen's d (raw and transformed)...")
    
    s_pooled_raw_weighted = compute_pooled_std(s_pos_raw, s_neg_raw, n_pos, n_neg, weighted=True)
    s_pooled_raw_simple = compute_pooled_std(s_pos_raw, s_neg_raw, n_pos, n_neg, weighted=False)
    
    s_pooled_stat_weighted = compute_pooled_std(s_pos_stat, s_neg_stat, n_pos, n_neg, weighted=True)
    s_pooled_stat_simple = compute_pooled_std(s_pos_stat, s_neg_stat, n_pos, n_neg, weighted=False)
    
    cohens_d_raw = compute_cohens_d(mean_pos_raw, mean_neg_raw, s_pooled_raw_weighted)
    cohens_d_raw_simple = compute_cohens_d(mean_pos_raw, mean_neg_raw, s_pooled_raw_simple)
    
    cohens_d_stat = compute_cohens_d(mean_pos_stat, mean_neg_stat, s_pooled_stat_weighted)
    
    # ------------------------------------------------------------
    # Hedges' g (small-sample correction)
    # ------------------------------------------------------------
    hedges_g_raw, hedges_correction = compute_hedges_g(cohens_d_raw, n_total)
    hedges_g_stat, _ = compute_hedges_g(cohens_d_stat, n_total)
    
    print(f"  Hedges correction factor (N={n_total}): {hedges_correction:.4f}")
    
    # ------------------------------------------------------------
    # Glass's Δ
    # ------------------------------------------------------------
    if do_glass:
        glass_delta_neg_ref, glass_delta_pos_ref = compute_glass_delta(
            mean_pos_raw, mean_neg_raw, s_pos_raw, s_neg_raw,
        )
    else:
        glass_delta_neg_ref = np.zeros(sae_dim)
        glass_delta_pos_ref = np.zeros(sae_dim)
    
    # ------------------------------------------------------------
    # Robust effect size (median + MAD)
    # ------------------------------------------------------------
    if do_robust:
        print("  Computing robust effect size (median/MAD)...")
        d_robust_raw = compute_robust_effect_sizes_vectorized(Z_pos, Z_neg)
    else:
        d_robust_raw = np.zeros(sae_dim)
    
    # ------------------------------------------------------------
    # Welch's t-test (significance for d)
    # ------------------------------------------------------------
    print("  Computing Welch's t-test p-values...")
    t_stat_raw, df_welch_raw, p_value_raw = welch_t_test_vectorized(Z_pos, Z_neg)
    t_stat_stat, df_welch_stat, p_value_stat = welch_t_test_vectorized(Z_pos_stat, Z_neg_stat)
    
    # FDR-corrected q-values
    q_value_raw = benjamini_hochberg(p_value_raw)
    q_value_stat = benjamini_hochberg(p_value_stat)
    
    # ------------------------------------------------------------
    # Analytical CI for Cohen's d (raw)
    # ------------------------------------------------------------
    print("  Computing analytical CIs for Cohen's d...")
    d_ci_lower_analytical, d_ci_upper_analytical, d_se_analytical = compute_d_confidence_interval(
        cohens_d_raw, n_pos, n_neg, alpha=0.05,
    )
    
    # ------------------------------------------------------------
    # Bootstrap CI (only on active features for speed)
    # ------------------------------------------------------------
    d_boot_mean = np.zeros(sae_dim)
    d_boot_std = np.zeros(sae_dim)
    d_boot_ci_lower = np.zeros(sae_dim)
    d_boot_ci_upper = np.zeros(sae_dim)
    
    if do_bootstrap:
        Z_pos_active = Z_pos[:, active_feature_ids]
        Z_neg_active = Z_neg[:, active_feature_ids]
        boot = bootstrap_cohens_d(
            Z_pos_active, Z_neg_active,
            n_bootstrap=n_bootstrap, alpha=0.05,
            random_state=random_state, n_jobs=n_jobs,
        )
        d_boot_mean[active_feature_ids] = boot["d_boot_mean"]
        d_boot_std[active_feature_ids] = boot["d_boot_std"]
        d_boot_ci_lower[active_feature_ids] = boot["d_boot_ci_lower"]
        d_boot_ci_upper[active_feature_ids] = boot["d_boot_ci_upper"]
    
    # ------------------------------------------------------------
    # Interpretive metrics
    # ------------------------------------------------------------
    overlap_coef = compute_overlap_coefficient(cohens_d_raw)
    prob_superiority = compute_probability_of_superiority(cohens_d_raw)
    
    # Effect size category (Cohen's conventions)
    abs_d = np.abs(cohens_d_raw)
    effect_category = np.where(
        abs_d < 0.2, "trivial",
        np.where(abs_d < 0.5, "small",
            np.where(abs_d < 0.8, "medium",
                np.where(abs_d < 1.5, "large",
                    np.where(abs_d < 3.0, "very_large", "massive")))))
    
    # ------------------------------------------------------------
    # Reliability flag: CI excludes 0 (effect-size analog of significance)
    # ------------------------------------------------------------
    ci_excludes_zero_analytical = (
        (d_ci_lower_analytical > 0) | (d_ci_upper_analytical < 0)
    )
    if do_bootstrap:
        ci_excludes_zero_bootstrap = (
            (d_boot_ci_lower > 0) | (d_boot_ci_upper < 0)
        )
    else:
        ci_excludes_zero_bootstrap = ci_excludes_zero_analytical
    
    # ------------------------------------------------------------
    # Robustness flag: raw d agrees with robust d in sign and magnitude
    # ------------------------------------------------------------
    # If robust d disagrees in sign, outliers are dominating
    sign_agrees_with_robust = (
        np.sign(cohens_d_raw) == np.sign(d_robust_raw)
    ) | (np.abs(cohens_d_raw) < 0.1)  # tolerate sign disagreement for tiny effects
    
    # ------------------------------------------------------------
    # Compose master DataFrame
    # ------------------------------------------------------------
    results = pd.DataFrame({
        "layer": layer,
        "feature_id": np.arange(sae_dim),
        # Primary effect size metrics
        "cohens_d_raw": cohens_d_raw,
        "cohens_d_stat": cohens_d_stat,  # transformed
        "abs_cohens_d_raw": np.abs(cohens_d_raw),
        "hedges_g_raw": hedges_g_raw,
        "hedges_g_stat": hedges_g_stat,
        "cohens_d_simple_pooled": cohens_d_raw_simple,
        # Glass's Δ
        "glass_delta_neg_ref": glass_delta_neg_ref,
        "glass_delta_pos_ref": glass_delta_pos_ref,
        # Robust effect size
        "d_robust_raw": d_robust_raw,
        "sign_agrees_with_robust": sign_agrees_with_robust,
        # Significance
        "t_stat_raw": t_stat_raw,
        "df_welch_raw": df_welch_raw,
        "p_value_raw": p_value_raw,
        "q_value_raw_bh": q_value_raw,
        "p_value_stat": p_value_stat,
        "q_value_stat_bh": q_value_stat,
        # Analytical CI
        "d_ci_lower_analytical": d_ci_lower_analytical,
        "d_ci_upper_analytical": d_ci_upper_analytical,
        "d_se_analytical": d_se_analytical,
        "ci_excludes_zero_analytical": ci_excludes_zero_analytical,
        # Bootstrap CI
        "d_boot_mean": d_boot_mean,
        "d_boot_std": d_boot_std,
        "d_boot_ci_lower": d_boot_ci_lower,
        "d_boot_ci_upper": d_boot_ci_upper,
        "ci_excludes_zero_bootstrap": ci_excludes_zero_bootstrap,
        # Interpretive
        "overlap_coefficient": overlap_coef,
        "probability_of_superiority": prob_superiority,
        "effect_category": effect_category,
        "direction_raw": direction,
        # Activity
        "is_active_feature": active_mask,
        "active_count_all": active_count_all,
        "activity_rate_all": activity_rate_all,
        "activity_rate_pos": activity_rate_pos,
        "activity_rate_neg": activity_rate_neg,
        # Descriptive (raw)
        "mean_positive_raw": mean_pos_raw,
        "mean_negative_raw": mean_neg_raw,
        "std_positive_raw": s_pos_raw,
        "std_negative_raw": s_neg_raw,
        "contrast_pos_minus_neg_raw": contrast_raw,
        "transform_used": transform,
    })
    
    # ------------------------------------------------------------
    # Ranking by |Cohen's d|
    # ------------------------------------------------------------
    # We rank by absolute value because direction is captured separately
    order = np.argsort(np.abs(cohens_d_raw))[::-1]
    results = results.iloc[order].reset_index(drop=True)
    results["d_rank"] = np.arange(1, len(results) + 1)
    
    cols = results.columns.tolist()
    cols.remove("d_rank")
    cols = ["d_rank"] + cols
    results = results[cols]
    
    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    summary = {
        "layer": int(layer),
        "num_rows": int(n_rows),
        "num_positive": int(n_pos),
        "num_negative": int(n_neg),
        "label_1_name": LABEL_1_NAME,
        "label_0_name": LABEL_0_NAME,
        "sae_dim": int(sae_dim),
        "transform": transform,
        "min_activity_rate": float(min_activity_rate),
        "min_active_count": int(min_active_count),
        "num_active_features": int(active_mask.sum()),
        "hedges_correction_factor": float(hedges_correction) if not np.isnan(hedges_correction) else None,
        "n_bootstraps": int(n_bootstrap) if do_bootstrap else 0,
        # Top feature
        "top_feature_id": int(results.iloc[0]["feature_id"]),
        "top_feature_d": float(results.iloc[0]["cohens_d_raw"]),
        "top_feature_direction": str(results.iloc[0]["direction_raw"]),
        # Effect size distribution
        "num_features_d_gt_0_2": int(np.sum(np.abs(cohens_d_raw) > 0.2)),
        "num_features_d_gt_0_5": int(np.sum(np.abs(cohens_d_raw) > 0.5)),
        "num_features_d_gt_0_8": int(np.sum(np.abs(cohens_d_raw) > 0.8)),
        "num_features_d_gt_1_5": int(np.sum(np.abs(cohens_d_raw) > 1.5)),
        "num_features_d_gt_3_0": int(np.sum(np.abs(cohens_d_raw) > 3.0)),
        "num_pos_direction": int(np.sum(cohens_d_raw > 0)),
        "num_neg_direction": int(np.sum(cohens_d_raw < 0)),
        # Significance
        "num_features_q_less_0_05": int(np.sum(q_value_raw < 0.05)),
        "num_features_q_less_0_01": int(np.sum(q_value_raw < 0.01)),
        # CI reliability
        "num_ci_excludes_zero_analytical": int(np.sum(ci_excludes_zero_analytical)),
        "num_ci_excludes_zero_bootstrap": int(np.sum(ci_excludes_zero_bootstrap)) if do_bootstrap else 0,
        # Robustness
        "num_sign_agrees_with_robust": int(np.sum(sign_agrees_with_robust)),
    }
    
    return results, summary


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================
def save_plots(results, Z, y, layer, out_dir, top_n_dist=10):
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)
    
    # Plot 1: Cohen's d histogram (signed)
    plt.figure(figsize=(10, 6))
    plt.hist(results["cohens_d_raw"].values, bins=100)
    plt.axvline(0, color='black', linestyle='-', alpha=0.5)
    for thresh, color in [(0.2, 'gray'), (0.5, 'orange'), (0.8, 'red'), (1.5, 'darkred')]:
        plt.axvline(thresh, color=color, linestyle='--', alpha=0.5, label=f'|d|={thresh}')
        plt.axvline(-thresh, color=color, linestyle='--', alpha=0.5)
    plt.xlabel("Cohen's d (signed)")
    plt.ylabel("Number of features")
    plt.title(f"Layer {layer}: Cohen's d distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"layer_{layer}_d_histogram.png"), dpi=150)
    plt.close()
    
    # Plot 2: |d| vs Hedges' g (should be nearly linear)
    plt.figure(figsize=(10, 6))
    plt.scatter(
        np.abs(results["cohens_d_raw"]),
        np.abs(results["hedges_g_raw"]),
        s=2, alpha=0.4,
    )
    max_val = np.abs(results["cohens_d_raw"]).max()
    plt.plot([0, max_val], [0, max_val], 'r--', label='y = x')
    plt.xlabel("|Cohen's d|")
    plt.ylabel("|Hedges' g|")
    plt.title(f"Layer {layer}: d vs Hedges' g (correction effect)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"layer_{layer}_d_vs_hedges_g.png"), dpi=150)
    plt.close()
    
    # Plot 3: Cohen's d vs robust d (outlier detection)
    plt.figure(figsize=(10, 6))
    plt.scatter(
        results["cohens_d_raw"],
        results["d_robust_raw"],
        s=2, alpha=0.4,
    )
    lim = max(np.abs(results["cohens_d_raw"]).max(), np.abs(results["d_robust_raw"]).max())
    plt.plot([-lim, lim], [-lim, lim], 'r--', label='y = x')
    plt.axhline(0, color='black', alpha=0.3)
    plt.axvline(0, color='black', alpha=0.3)
    plt.xlabel("Cohen's d")
    plt.ylabel("Robust d (median/MAD)")
    plt.title(f"Layer {layer}: Cohen's d vs robust d (outlier check)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"layer_{layer}_d_vs_robust.png"), dpi=150)
    plt.close()
    
    # Plot 4: Effect size category counts (bar chart)
    cat_counts = results["effect_category"].value_counts().reindex(
        ["trivial", "small", "medium", "large", "very_large", "massive"], fill_value=0
    )
    plt.figure(figsize=(10, 6))
    plt.bar(cat_counts.index, cat_counts.values)
    plt.xlabel("Effect size category")
    plt.ylabel("Number of features")
    plt.title(f"Layer {layer}: Effect size distribution")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"layer_{layer}_effect_categories.png"), dpi=150)
    plt.close()
    
    # Plot 5: Top 50 |d| with bootstrap CI error bars
    top50 = results.head(50)
    plt.figure(figsize=(14, 6))
    x = np.arange(len(top50))
    d_vals = top50["cohens_d_raw"].values
    ci_low = top50["d_boot_ci_lower"].values
    ci_high = top50["d_boot_ci_upper"].values
    err_low = d_vals - ci_low
    err_high = ci_high - d_vals
    plt.errorbar(
        x, d_vals,
        yerr=[np.abs(err_low), np.abs(err_high)],
        fmt='o', markersize=4, capsize=2, alpha=0.7,
    )
    plt.axhline(0, color='black', alpha=0.5)
    plt.xlabel("Top feature rank")
    plt.ylabel("Cohen's d (with 95% bootstrap CI)")
    plt.title(f"Layer {layer}: Top 50 features by |d|")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"layer_{layer}_top50_d_with_ci.png"), dpi=150)
    plt.close()
    
    # Plot 6: Volcano plot: |d| vs -log10(p) — shows magnitude AND significance
    plt.figure(figsize=(10, 6))
    neg_log_p = -np.log10(np.maximum(results["p_value_raw"].values, 1e-300))
    plt.scatter(
        results["cohens_d_raw"], neg_log_p,
        s=2, alpha=0.4,
    )
    plt.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
    plt.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='|d|=0.5')
    plt.axvline(-0.5, color='orange', linestyle='--', alpha=0.5)
    plt.axvline(0.8, color='darkred', linestyle='--', alpha=0.5, label='|d|=0.8')
    plt.axvline(-0.8, color='darkred', linestyle='--', alpha=0.5)
    plt.xlabel("Cohen's d")
    plt.ylabel("-log10(p-value)")
    plt.title(f"Layer {layer}: Volcano plot (effect size vs significance)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"layer_{layer}_volcano.png"), dpi=150)
    plt.close()
    
    # Plot 7: Pos vs neg distributions for top features
    pos_mask = y == 1
    neg_mask = y == 0
    for _, row in results.head(top_n_dist).iterrows():
        fid = int(row["feature_id"])
        pos_vals = Z[pos_mask, fid]
        neg_vals = Z[neg_mask, fid]
        plt.figure(figsize=(10, 6))
        plt.hist(pos_vals, bins=50, alpha=0.6, label=LABEL_1_NAME)
        plt.hist(neg_vals, bins=50, alpha=0.6, label=LABEL_0_NAME)
        plt.xlabel(f"Raw activation of feature {fid}")
        plt.ylabel("Count")
        plt.title(
            f"Layer {layer}, feature {fid}, "
            f"d={row['cohens_d_raw']:.3f}, |g|={np.abs(row['hedges_g_raw']):.3f}, "
            f"category={row['effect_category']}, direction={row['direction_raw']}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(plot_dir, f"layer_{layer}_feature_{fid}_distribution.png"),
            dpi=150,
        )
        plt.close()


# ============================================================
# PER-LAYER RUNNER
# ============================================================
def run_layer(
    base_dir, out_dir, layer, transform,
    min_activity_rate, min_active_count,
    allowed_splits, top_n,
    do_bootstrap, n_bootstrap,
    do_robust, do_glass,
    n_jobs, random_state, make_plots,
):
    input_path, Z, y, metadata, keys = load_layer_file(base_dir, layer)
    Z_use, y_use, metadata_use, row_mask = filter_rows(
        Z, y, metadata, allowed_splits=allowed_splits,
    )
    
    print("\n" + "=" * 90)
    print(f"LAYER {layer}")
    print("=" * 90)
    print("Input:", input_path)
    print("Z shape:", Z.shape, " | Used:", Z_use.shape)
    print(f"Pos: {int(np.sum(y_use == 1))} | Neg: {int(np.sum(y_use == 0))}")
    print("Transform:", transform)
    
    if allowed_splits is None or len(allowed_splits) == 0:
        split_tag = "all_rows"
    else:
        split_tag = "_".join([str(s).strip().lower() for s in allowed_splits])
    
    results, summary = compute_complete_cohens_d(
        Z=Z_use, y=y_use, layer=layer,
        transform=transform,
        min_activity_rate=min_activity_rate,
        min_active_count=min_active_count,
        do_bootstrap=do_bootstrap,
        n_bootstrap=n_bootstrap,
        do_robust=do_robust,
        do_glass=do_glass,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    
    summary["input_path"] = input_path
    summary["rows_used"] = int(Z_use.shape[0])
    summary["row_filter"] = split_tag
    
    layer_out_dir = os.path.join(out_dir, f"layer_{layer}")
    ensure_dir(layer_out_dir)
    
    suffix = f"{split_tag}_{transform}"
    full_csv = os.path.join(layer_out_dir, f"layer_{layer}_d_full_{suffix}.csv")
    top_csv = os.path.join(layer_out_dir, f"layer_{layer}_d_top{top_n}_{suffix}.csv")
    top_npy = os.path.join(layer_out_dir, f"layer_{layer}_d_top{top_n}_ids_{suffix}.npy")
    summary_json = os.path.join(layer_out_dir, f"layer_{layer}_d_summary_{suffix}.json")
    scores_npz = os.path.join(layer_out_dir, f"layer_{layer}_d_scores_{suffix}.npz")
    
    results.to_csv(full_csv, index=False)
    results.head(top_n).to_csv(top_csv, index=False)
    np.save(top_npy, results.head(top_n)["feature_id"].astype(np.int32).values)
    safe_json_dump(summary, summary_json)
    
    # Save NPZ for fast downstream loading
    results_by_id = results.sort_values("feature_id")
    np.savez_compressed(
        scores_npz,
        feature_id=np.arange(summary["sae_dim"]),
        cohens_d_raw=results_by_id["cohens_d_raw"].values,
        cohens_d_stat=results_by_id["cohens_d_stat"].values,
        abs_cohens_d_raw=results_by_id["abs_cohens_d_raw"].values,
        hedges_g_raw=results_by_id["hedges_g_raw"].values,
        d_robust_raw=results_by_id["d_robust_raw"].values,
        glass_delta_neg_ref=results_by_id["glass_delta_neg_ref"].values,
        glass_delta_pos_ref=results_by_id["glass_delta_pos_ref"].values,
        p_value_raw=results_by_id["p_value_raw"].values,
        q_value_raw_bh=results_by_id["q_value_raw_bh"].values,
        d_ci_lower_analytical=results_by_id["d_ci_lower_analytical"].values,
        d_ci_upper_analytical=results_by_id["d_ci_upper_analytical"].values,
        d_boot_mean=results_by_id["d_boot_mean"].values,
        d_boot_std=results_by_id["d_boot_std"].values,
        d_boot_ci_lower=results_by_id["d_boot_ci_lower"].values,
        d_boot_ci_upper=results_by_id["d_boot_ci_upper"].values,
        overlap_coefficient=results_by_id["overlap_coefficient"].values,
        probability_of_superiority=results_by_id["probability_of_superiority"].values,
        ci_excludes_zero_analytical=results_by_id["ci_excludes_zero_analytical"].values,
        ci_excludes_zero_bootstrap=results_by_id["ci_excludes_zero_bootstrap"].values,
        sign_agrees_with_robust=results_by_id["sign_agrees_with_robust"].values,
        is_active_feature=results_by_id["is_active_feature"].values,
        activity_rate_all=results_by_id["activity_rate_all"].values,
        contrast_pos_minus_neg_raw=results_by_id["contrast_pos_minus_neg_raw"].values,
    )
    
    print("\nTop 20 features:")
    print(results.head(20)[
        ["d_rank", "feature_id", "cohens_d_raw", "hedges_g_raw",
         "d_robust_raw", "p_value_raw", "q_value_raw_bh",
         "d_boot_ci_lower", "d_boot_ci_upper",
         "effect_category", "direction_raw"]
    ].to_string(index=False))
    
    print(f"\nSaved:\n  {full_csv}\n  {top_csv}\n  {top_npy}\n  {scores_npz}\n  {summary_json}")
    
    if make_plots:
        save_plots(
            results=results, Z=Z_use, y=y_use, layer=layer, out_dir=layer_out_dir,
        )
        print("Plots saved under:", os.path.join(layer_out_dir, "plots"))
    
    return summary


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Complete Cohen's d effect size analysis for SAE features.",
    )
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--transform", choices=["raw", "log1p"], default="log1p")
    parser.add_argument("--min_activity_rate", type=float, default=0.01)
    parser.add_argument("--min_active_count", type=int, default=5)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--top_n", type=int, default=2000)
    parser.add_argument("--no_bootstrap", action="store_true")
    parser.add_argument("--n_bootstrap", type=int, default=500)
    parser.add_argument("--no_robust", action="store_true")
    parser.add_argument("--no_glass", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--make_plots", action="store_true")
    args = parser.parse_args()
    
    ensure_dir(args.out_dir)
    
    all_summaries = []
    for layer in args.layers:
        summary = run_layer(
            base_dir=args.base_dir, out_dir=args.out_dir, layer=layer,
            transform=args.transform,
            min_activity_rate=args.min_activity_rate,
            min_active_count=args.min_active_count,
            allowed_splits=args.splits, top_n=args.top_n,
            do_bootstrap=not args.no_bootstrap,
            n_bootstrap=args.n_bootstrap,
            do_robust=not args.no_robust,
            do_glass=not args.no_glass,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            make_plots=args.make_plots,
        )
        all_summaries.append(summary)
    
    summary_df = pd.DataFrame(all_summaries)
    summary_csv = os.path.join(args.out_dir, f"d_all_layers_summary_{args.transform}.csv")
    summary_json = os.path.join(args.out_dir, f"d_all_layers_summary_{args.transform}.json")
    summary_df.to_csv(summary_csv, index=False)
    safe_json_dump(all_summaries, summary_json)
    
    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print("All-layer summary:", summary_csv)


if __name__ == "__main__":
    main()
