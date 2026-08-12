import os
import json
import argparse
import warnings

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

from sklearn.feature_selection import f_classif


"""
Complete F-stat analysis for political polarity SAE features.

Politics labels in the extracted SAE files are:
    y = 1 -> right/conservative direction text
    y = 0 -> left/liberal direction text

This script mirrors the sentiment complete F-stat pipeline, but all
class-specific outputs use right/left names instead of
positive/negative.
"""


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_BASE_DIR = os.path.join(PROJECT_ROOT, "results", "sae_features_gemma3_4b")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "results", "fstat_complete")
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
        json.dump(obj, f, indent=2)


def value_counts(values):
    counts = {}
    for value in values:
        key = str(value).strip().lower()
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg FDR correction.

    Input:
        p_values: array of p-values

    Output:
        q_values: FDR-adjusted p-values
    """
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
    """
    transform:
        raw   -> use raw SAE activations
        log1p -> use log(1 + activation)
    """
    if transform == "raw":
        return Z

    if transform == "log1p":
        return np.log1p(Z)

    raise ValueError(f"Unknown transform: {transform}")


def pool_if_needed(Z, pool_method="max"):
    """
    Your current Z should be 2D:
        (num_texts, sae_dim)

    But if somehow Z is token-level:
        (num_texts, seq_len, sae_dim)

    then this converts it to sample-level.
    """
    if Z.ndim == 2:
        return Z

    if Z.ndim == 3:
        print(f"Detected 3D Z: {Z.shape}")

        if pool_method == "max":
            print("Applying max pooling over tokens.")
            return Z.max(axis=1)

        if pool_method == "mean":
            print("Applying mean pooling over tokens.")
            return Z.mean(axis=1)

        raise ValueError(f"Unknown pool_method: {pool_method}")

    raise ValueError(f"Unsupported Z shape: {Z.shape}")


def validate_loaded_data(Z, y):
    if len(Z) != len(y):
        raise ValueError(f"Z rows and y length mismatch: Z={Z.shape}, y={y.shape}")

    unique_labels = sorted(set(y.tolist()))
    if unique_labels != [0, 1]:
        raise ValueError(f"Expected y labels [0, 1], got {unique_labels}")

    n_right = int(np.sum(y == 1))
    n_left = int(np.sum(y == 0))

    if n_right == 0 or n_left == 0:
        raise ValueError("Need both right and left rows for F-stat.")

    if Z.shape[0] < 4:
        raise ValueError("Too few rows for F-stat.")

    if Z.shape[1] < 2:
        raise ValueError("Too few SAE features.")


# ============================================================
# LOADING
# ============================================================

def load_layer_file(base_dir, layer):
    path = os.path.join(base_dir, f"layer_{layer}_all_sae_features.npz")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    data = np.load(path, allow_pickle=True)

    required_keys = ["Z", "y"]
    for key in required_keys:
        if key not in data.files:
            raise KeyError(f"Missing key '{key}' in {path}. Found keys: {data.files}")

    Z = data["Z"]
    y = data["y"].astype(np.int32)

    Z = pool_if_needed(Z)
    Z = np.asarray(Z, dtype=np.float32)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    validate_loaded_data(Z, y)

    metadata = {}

    for key in [
        "pair_ids",
        "sides",
        "split",
        "split2",
        "topic",
        "category",
        "target",
        "sources",
    ]:
        if key in data.files:
            metadata[key] = data[key]

    return path, Z, y, metadata, data.files


def filter_rows(Z, y, metadata, allowed_splits=None):
    """
    By default, use all rows.

    If allowed_splits is provided, use only those split2 values.
    """
    if allowed_splits is None or len(allowed_splits) == 0:
        row_mask = np.ones(len(y), dtype=bool)
        return Z, y, metadata, row_mask

    if "split2" not in metadata:
        raise ValueError("allowed_splits was provided, but split2 is not in metadata.")

    split2 = as_str_array(metadata["split2"])
    allowed = set([str(s).strip().lower() for s in allowed_splits])

    row_mask = np.array([s in allowed for s in split2], dtype=bool)

    if row_mask.sum() == 0:
        raise ValueError(f"No rows matched allowed_splits={allowed_splits}")

    Z_use = Z[row_mask]
    y_use = y[row_mask]

    metadata_use = {}
    for key, value in metadata.items():
        if len(value) == len(row_mask):
            metadata_use[key] = value[row_mask]
        else:
            metadata_use[key] = value

    validate_loaded_data(Z_use, y_use)

    return Z_use, y_use, metadata_use, row_mask


# ============================================================
# MAIN F-STAT ANALYSIS
# ============================================================

def compute_complete_fstat(
    Z,
    y,
    layer,
    transform="log1p",
    min_activity_rate=0.0,
    min_active_count=1,
):
    """
    Complete F-stat analysis.

    F-stat is computed on transformed activations:
        raw or log1p

    Direction and descriptive stats are computed on raw activations.
    """
    n_rows, sae_dim = Z.shape

    right_mask = y == 1
    left_mask = y == 0

    Z_right = Z[right_mask]
    Z_left = Z[left_mask]

    n_right = Z_right.shape[0]
    n_left = Z_left.shape[0]

    # ------------------------------------------------------------
    # Raw descriptive statistics
    # ------------------------------------------------------------

    active_count_all = np.count_nonzero(Z > 0, axis=0)
    active_count_right = np.count_nonzero(Z_right > 0, axis=0)
    active_count_left = np.count_nonzero(Z_left > 0, axis=0)

    activity_rate_all = active_count_all / n_rows
    activity_rate_right = active_count_right / n_right
    activity_rate_left = active_count_left / n_left

    mean_all_raw = Z.mean(axis=0)
    mean_right_raw = Z_right.mean(axis=0)
    mean_left_raw = Z_left.mean(axis=0)

    std_all_raw = Z.std(axis=0, ddof=1)
    std_right_raw = Z_right.std(axis=0, ddof=1)
    std_left_raw = Z_left.std(axis=0, ddof=1)

    var_all_raw = Z.var(axis=0, ddof=1)

    contrast_raw = mean_right_raw - mean_left_raw

    direction = np.where(
        contrast_raw > 0,
        LABEL_1_NAME,
        np.where(contrast_raw < 0, LABEL_0_NAME, "neutral")
    )

    # ------------------------------------------------------------
    # Feature filter for stable F-stat
    # ------------------------------------------------------------

    active_mask = (
        (activity_rate_all >= min_activity_rate)
        & (active_count_all >= min_active_count)
        & (var_all_raw > 1e-12)
    )

    active_feature_ids = np.where(active_mask)[0]

    # Initialize full-size outputs
    f_scores = np.zeros(sae_dim, dtype=np.float64)
    p_values = np.ones(sae_dim, dtype=np.float64)

    mean_all_stat = np.zeros(sae_dim, dtype=np.float64)
    mean_right_stat = np.zeros(sae_dim, dtype=np.float64)
    mean_left_stat = np.zeros(sae_dim, dtype=np.float64)
    contrast_stat = np.zeros(sae_dim, dtype=np.float64)

    # ------------------------------------------------------------
    # Transform activations for F-stat
    # ------------------------------------------------------------

    Z_stat = apply_transform(Z, transform)

    Z_right_stat = Z_stat[right_mask]
    Z_left_stat = Z_stat[left_mask]

    mean_all_stat[:] = Z_stat.mean(axis=0)
    mean_right_stat[:] = Z_right_stat.mean(axis=0)
    mean_left_stat[:] = Z_left_stat.mean(axis=0)
    contrast_stat[:] = mean_right_stat - mean_left_stat

    # ------------------------------------------------------------
    # F-statistic
    # ------------------------------------------------------------

    if len(active_feature_ids) == 0:
        raise ValueError("No active features survived filtering.")

    X_active = Z_stat[:, active_feature_ids]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_active, p_active = f_classif(X_active, y)

    f_active = np.nan_to_num(f_active, nan=0.0, posinf=0.0, neginf=0.0)
    p_active = np.nan_to_num(p_active, nan=1.0, posinf=1.0, neginf=1.0)

    f_scores[active_feature_ids] = f_active
    p_values[active_feature_ids] = p_active

    # ------------------------------------------------------------
    # FDR q-values
    # ------------------------------------------------------------

    q_values = benjamini_hochberg(p_values)

    # ------------------------------------------------------------
    # ANOVA effect size: eta squared
    # For two classes, df_between = 1, df_within = n_right + n_left - 2
    # eta^2 = SS_between / SS_total
    # Can be computed from F:
    # eta^2 = (F * df_between) / (F * df_between + df_within)
    # ------------------------------------------------------------

    df_between = 1
    df_within = n_right + n_left - 2

    eta_squared = (f_scores * df_between) / (
        (f_scores * df_between) + df_within + 1e-12
    )

    # ------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------

    order = np.argsort(f_scores)[::-1]

    results = pd.DataFrame({
        "layer": layer,
        "feature_id": np.arange(sae_dim),

        # Main F-stat outputs
        "f_score": f_scores,
        "p_value": p_values,
        "q_value_bh": q_values,
        "eta_squared": eta_squared,

        # Activity information
        "is_active_feature": active_mask,
        "active_count_all": active_count_all,
        "active_count_right": active_count_right,
        "active_count_left": active_count_left,
        "activity_rate_all": activity_rate_all,
        "activity_rate_right": activity_rate_right,
        "activity_rate_left": activity_rate_left,

        # Raw activation descriptive statistics
        "mean_all_raw": mean_all_raw,
        "mean_right_raw": mean_right_raw,
        "mean_left_raw": mean_left_raw,
        "std_all_raw": std_all_raw,
        "std_right_raw": std_right_raw,
        "std_left_raw": std_left_raw,
        "contrast_right_minus_left_raw": contrast_raw,
        "direction_raw": direction,

        # Transformed activation statistics
        "mean_all_stat_input": mean_all_stat,
        "mean_right_stat_input": mean_right_stat,
        "mean_left_stat_input": mean_left_stat,
        "contrast_right_minus_left_stat_input": contrast_stat,
        "transform_used_for_fstat": transform,
    })

    results = results.iloc[order].reset_index(drop=True)
    results["f_rank"] = np.arange(1, len(results) + 1)

    # Move rank near front
    cols = results.columns.tolist()
    cols.remove("f_rank")
    cols = ["f_rank"] + cols
    results = results[cols]

    summary = {
        "layer": int(layer),
        "num_rows": int(n_rows),
        "num_right": int(n_right),
        "num_left": int(n_left),
        "label_1_name": LABEL_1_NAME,
        "label_0_name": LABEL_0_NAME,
        "sae_dim": int(sae_dim),
        "transform": transform,
        "min_activity_rate": float(min_activity_rate),
        "min_active_count": int(min_active_count),
        "num_active_features": int(active_mask.sum()),
        "mean_l0_per_row": float(np.count_nonzero(Z > 0, axis=1).mean()),
        "median_l0_per_row": float(np.median(np.count_nonzero(Z > 0, axis=1))),
        "mean_activity_rate_all_features": float(activity_rate_all.mean()),
        "max_f_score": float(results.iloc[0]["f_score"]),
        "top_feature_id": int(results.iloc[0]["feature_id"]),
        "top_feature_direction": str(results.iloc[0]["direction_raw"]),
        "num_features_q_less_0_05": int(np.sum(q_values < 0.05)),
        "num_features_q_less_0_01": int(np.sum(q_values < 0.01)),
        "num_features_q_less_0_001": int(np.sum(q_values < 0.001)),
        "df_between": int(df_between),
        "df_within": int(df_within),
    }

    return results, summary


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

def save_plots(results, Z, y, layer, out_dir, top_n_features_for_distribution=10):
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)

    # ------------------------------------------------------------
    # Plot 1: F-score histogram
    # ------------------------------------------------------------

    plt.figure(figsize=(10, 6))
    plt.hist(results["f_score"].values, bins=100)
    plt.xlabel("F-score")
    plt.ylabel("Number of SAE features")
    plt.title(f"Layer {layer}: F-score distribution")
    plt.tight_layout()

    path = os.path.join(plot_dir, f"layer_{layer}_fscore_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # Plot 2: Top 50 F-scores
    # ------------------------------------------------------------

    top50 = results.head(50)

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(top50)), top50["f_score"].values)
    plt.xlabel("Top feature rank")
    plt.ylabel("F-score")
    plt.title(f"Layer {layer}: Top 50 F-stat features")
    plt.tight_layout()

    path = os.path.join(plot_dir, f"layer_{layer}_top50_fscores.png")
    plt.savefig(path, dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # Plot 3: Right vs left distributions for top features
    # ------------------------------------------------------------

    right_mask = y == 1
    left_mask = y == 0

    for _, row in results.head(top_n_features_for_distribution).iterrows():
        fid = int(row["feature_id"])

        right_vals = Z[right_mask, fid]
        left_vals = Z[left_mask, fid]

        plt.figure(figsize=(10, 6))
        plt.hist(right_vals, bins=50, alpha=0.6, label=LABEL_1_NAME)
        plt.hist(left_vals, bins=50, alpha=0.6, label=LABEL_0_NAME)
        plt.xlabel(f"Raw activation of feature {fid}")
        plt.ylabel("Count")
        plt.title(
            f"Layer {layer}, feature {fid}, "
            f"F={row['f_score']:.3f}, direction={row['direction_raw']}"
        )
        plt.legend()
        plt.tight_layout()

        path = os.path.join(
            plot_dir,
            f"layer_{layer}_feature_{fid}_right_vs_left_distribution.png",
        )
        plt.savefig(path, dpi=150)
        plt.close()


# ============================================================
# PER-LAYER RUNNER
# ============================================================

def run_layer(
    base_dir,
    out_dir,
    layer,
    transform,
    min_activity_rate,
    min_active_count,
    allowed_splits,
    top_n,
    make_plots,
):
    input_path, Z, y, metadata, keys = load_layer_file(base_dir, layer)

    Z_use, y_use, metadata_use, row_mask = filter_rows(
        Z,
        y,
        metadata,
        allowed_splits=allowed_splits,
    )

    print("\n" + "=" * 90)
    print(f"LAYER {layer}")
    print("=" * 90)
    print("Input file:", input_path)
    print("Keys:", list(keys))
    print("Original Z shape:", Z.shape)
    print("Used Z shape:", Z_use.shape)
    print("Right rows used:", int(np.sum(y_use == 1)))
    print("Left rows used:", int(np.sum(y_use == 0)))
    print("Transform for F-stat:", transform)
    print("Min activity rate:", min_activity_rate)
    print("Min active count:", min_active_count)

    if allowed_splits is None or len(allowed_splits) == 0:
        split_tag = "all_rows"
    else:
        split_tag = "_".join([str(s).strip().lower() for s in allowed_splits])

    results, summary = compute_complete_fstat(
        Z=Z_use,
        y=y_use,
        layer=layer,
        transform=transform,
        min_activity_rate=min_activity_rate,
        min_active_count=min_active_count,
    )

    summary["input_path"] = input_path
    summary["rows_used"] = int(Z_use.shape[0])
    summary["row_filter"] = split_tag
    if "split" in metadata_use:
        summary["split_counts_used"] = value_counts(metadata_use["split"])
    if "split2" in metadata_use:
        summary["split2_counts_used"] = value_counts(metadata_use["split2"])
    if "topic" in metadata_use:
        summary["num_topics_used"] = int(len(set(as_str_array(metadata_use["topic"]))))
    if "category" in metadata_use:
        summary["num_categories_used"] = int(len(set(as_str_array(metadata_use["category"]))))

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------

    layer_out_dir = os.path.join(out_dir, f"layer_{layer}")
    ensure_dir(layer_out_dir)

    full_csv = os.path.join(
        layer_out_dir,
        f"layer_{layer}_fstat_full_{split_tag}_{transform}.csv",
    )

    top_csv = os.path.join(
        layer_out_dir,
        f"layer_{layer}_fstat_top{top_n}_{split_tag}_{transform}.csv",
    )

    top_npy = os.path.join(
        layer_out_dir,
        f"layer_{layer}_top{top_n}_feature_ids_{split_tag}_{transform}.npy",
    )

    summary_json = os.path.join(
        layer_out_dir,
        f"layer_{layer}_fstat_summary_{split_tag}_{transform}.json",
    )

    results.to_csv(full_csv, index=False)
    results.head(top_n).to_csv(top_csv, index=False)
    np.save(top_npy, results.head(top_n)["feature_id"].astype(np.int32).values)
    safe_json_dump(summary, summary_json)

    print("\nTop 20 features:")
    print(
        results.head(20)[
            [
                "f_rank",
                "feature_id",
                "f_score",
                "p_value",
                "q_value_bh",
                "eta_squared",
                "activity_rate_all",
                "mean_right_raw",
                "mean_left_raw",
                "contrast_right_minus_left_raw",
                "direction_raw",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print("Full CSV:   ", full_csv)
    print("Top CSV:    ", top_csv)
    print("Top NPY:    ", top_npy)
    print("Summary:    ", summary_json)

    if make_plots:
        save_plots(
            results=results,
            Z=Z_use,
            y=y_use,
            layer=layer,
            out_dir=layer_out_dir,
            top_n_features_for_distribution=10,
        )
        print("Plots saved under:", os.path.join(layer_out_dir, "plots"))

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Complete F-stat analysis for political polarity SAE feature .npz "
            "files, with y=1 as right and y=0 as left."
        )
    )

    parser.add_argument(
        "--base_dir",
        default=DEFAULT_BASE_DIR,
        help="Directory containing layer_{layer}_all_sae_features.npz files.",
    )

    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory for F-stat results.",
    )

    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=DEFAULT_LAYERS,
        help="Layers to process.",
    )

    parser.add_argument(
        "--transform",
        choices=["raw", "log1p"],
        default="log1p",
        help="Activation transform used for F-stat. Recommended: log1p.",
    )

    parser.add_argument(
        "--min_activity_rate",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of rows where a feature must be active. "
            "Use 0.0 for no rate filter. Use 0.01 for stricter filtering."
        ),
    )

    parser.add_argument(
        "--min_active_count",
        type=int,
        default=1,
        help=(
            "Minimum number of rows where feature must be active. "
            "Recommended first run: 1. Stricter run: 5 or 10."
        ),
    )

    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help=(
            "Optional split2 values to use. "
            "Default: use all rows. Example: --splits probe attribution"
        ),
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=500,
        help="Number of top features to save separately.",
    )

    parser.add_argument(
        "--make_plots",
        action="store_true",
        help="Save diagnostic plots.",
    )

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    all_summaries = []

    for layer in args.layers:
        summary = run_layer(
            base_dir=args.base_dir,
            out_dir=args.out_dir,
            layer=layer,
            transform=args.transform,
            min_activity_rate=args.min_activity_rate,
            min_active_count=args.min_active_count,
            allowed_splits=args.splits,
            top_n=args.top_n,
            make_plots=args.make_plots,
        )

        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)

    summary_csv = os.path.join(
        args.out_dir,
        f"fstat_all_layers_summary_{args.transform}.csv",
    )

    summary_json = os.path.join(
        args.out_dir,
        f"fstat_all_layers_summary_{args.transform}.json",
    )

    summary_df.to_csv(summary_csv, index=False)
    safe_json_dump(all_summaries, summary_json)

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print("All-layer summary CSV:", summary_csv)
    print("All-layer summary JSON:", summary_json)
    print("Output directory:", args.out_dir)


if __name__ == "__main__":
    main()
