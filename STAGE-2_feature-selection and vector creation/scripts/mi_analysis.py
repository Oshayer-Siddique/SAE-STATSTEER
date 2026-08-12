"""
mi_analysis.py

Complete Mutual Information analysis for political polarity SAE features.
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
from sklearn.feature_selection import mutual_info_classif
from joblib import Parallel, delayed


DEFAULT_BASE_DIR = os.path.join(PROJECT_ROOT, "results", "sae_features_gemma3_4b")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "results", "mi_complete")
DEFAULT_LAYERS = [12, 19, 26, 31, 38]
LABEL_1_NAME = "right"
LABEL_0_NAME = "left"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def as_str_array(x):
    return np.array([str(v).strip().lower() for v in x])


def safe_json_dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


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
    n_right = int(np.sum(y == 1))
    n_left = int(np.sum(y == 0))
    if n_right == 0 or n_left == 0:
        raise ValueError("Need both right and left rows for MI.")
    if Z.shape[0] < 10:
        raise ValueError("Too few rows for reliable MI estimation (need >=10).")
    if Z.shape[1] < 2:
        raise ValueError("Too few SAE features.")


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
    if allowed_splits is None or len(allowed_splits) == 0:
        row_mask = np.ones(len(y), dtype=bool)
        return Z, y, metadata, row_mask
    if "split2" not in metadata:
        raise ValueError("allowed_splits provided but split2 missing from metadata.")
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


def compute_class_entropy(y):
    p_pos = np.mean(y == 1)
    p_neg = np.mean(y == 0)
    h = 0.0
    for p in [p_pos, p_neg]:
        if p > 0:
            h -= p * np.log(p)
    return h


def compute_mi_with_bootstrap(
    X,
    y,
    n_bootstrap=20,
    n_neighbors=3,
    subsample_frac=0.8,
    random_state=42,
    n_jobs=-1,
):
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    sub_size = int(subsample_frac * n_samples)

    def single_bootstrap(seed):
        local_rng = np.random.default_rng(seed)
        idx = local_rng.choice(n_samples, size=sub_size, replace=True)
        X_sub = X[idx]
        y_sub = y[idx]
        if np.sum(y_sub == 1) < 2 or np.sum(y_sub == 0) < 2:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return mutual_info_classif(
                X_sub, y_sub,
                discrete_features=False,
                n_neighbors=n_neighbors,
                random_state=int(seed),
            )

    seeds = rng.integers(0, 10**9, size=n_bootstrap)
    print(f"  Running {n_bootstrap} bootstrap iterations (n_jobs={n_jobs})...")
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(single_bootstrap)(seed) for seed in seeds
    )
    print(f"  Bootstrap completed in {time.time() - t0:.1f}s")

    valid_results = [r for r in results if r is not None]
    if len(valid_results) < n_bootstrap // 2:
        warnings.warn(f"Only {len(valid_results)}/{n_bootstrap} bootstraps succeeded.")

    mi_matrix = np.stack(valid_results, axis=0)
    return {
        "mi_mean": mi_matrix.mean(axis=0),
        "mi_std": mi_matrix.std(axis=0, ddof=1),
        "mi_q05": np.percentile(mi_matrix, 5, axis=0),
        "mi_q95": np.percentile(mi_matrix, 95, axis=0),
        "n_successful_bootstraps": len(valid_results),
    }


def compute_mi_permutation_pvalue(
    X,
    y,
    observed_mi,
    n_permutations=100,
    n_neighbors=3,
    random_state=42,
    n_jobs=-1,
):
    rng = np.random.default_rng(random_state)

    def single_permutation(seed):
        local_rng = np.random.default_rng(seed)
        y_perm = local_rng.permutation(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return mutual_info_classif(
                X, y_perm,
                discrete_features=False,
                n_neighbors=n_neighbors,
                random_state=int(seed),
            )

    seeds = rng.integers(0, 10**9, size=n_permutations)
    print(f"  Running {n_permutations} permutations for null distribution...")
    t0 = time.time()
    null_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(single_permutation)(seed) for seed in seeds
    )
    print(f"  Permutation test completed in {time.time() - t0:.1f}s")

    null_matrix = np.stack(null_results, axis=0)
    p_values = (np.sum(null_matrix >= observed_mi[None, :], axis=0) + 1) / (n_permutations + 1)
    null_mean = null_matrix.mean(axis=0)
    null_std = null_matrix.std(axis=0, ddof=1)
    mi_debiased = np.maximum(observed_mi - null_mean, 0.0)
    mi_zscore = (observed_mi - null_mean) / (null_std + 1e-12)
    return {
        "p_value_perm": p_values,
        "null_mean": null_mean,
        "mi_debiased": mi_debiased,
        "mi_zscore": mi_zscore,
    }


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


def compute_mi_multi_k(X, y, k_values=(3, 5, 10), random_state=42):
    print(f"  Computing MI at k={list(k_values)}...")
    mi_per_k = []
    for k in k_values:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi = mutual_info_classif(
                X, y,
                discrete_features=False,
                n_neighbors=k,
                random_state=random_state,
            )
        mi_per_k.append(mi)
    mi_matrix = np.stack(mi_per_k, axis=0)
    return {
        "mi_mean_over_k": mi_matrix.mean(axis=0),
        "mi_std_over_k": mi_matrix.std(axis=0, ddof=1),
        "mi_per_k": {f"mi_k{k}": mi_matrix[i] for i, k in enumerate(k_values)},
    }


def compute_class_conditional_stats(Z_raw, y):
    right_mask = y == 1
    left_mask = y == 0
    Z_right = Z_raw[right_mask]
    Z_left = Z_raw[left_mask]
    return {
        "mean_right_raw": Z_right.mean(axis=0),
        "mean_left_raw": Z_left.mean(axis=0),
        "activity_rate_right": (Z_right > 0).mean(axis=0),
        "activity_rate_left": (Z_left > 0).mean(axis=0),
        "activity_rate_all": (Z_raw > 0).mean(axis=0),
        "active_count_all": np.count_nonzero(Z_raw > 0, axis=0),
        "contrast_raw": Z_right.mean(axis=0) - Z_left.mean(axis=0),
        "var_all_raw": Z_raw.var(axis=0, ddof=1),
    }


def compute_complete_mi(
    Z,
    y,
    layer,
    transform="log1p",
    min_activity_rate=0.01,
    min_active_count=5,
    n_neighbors=3,
    multi_k=(3, 5, 10),
    n_bootstrap=20,
    n_permutations=100,
    do_bootstrap=True,
    do_permutation=True,
    n_jobs=-1,
    random_state=42,
):
    n_rows, sae_dim = Z.shape
    n_right = int(np.sum(y == 1))
    n_left = int(np.sum(y == 0))
    print(f"  Data: {n_rows} rows, {sae_dim} features, {n_right} right, {n_left} left")
    h_y = compute_class_entropy(y)
    print(f"  H(Y) = {h_y:.4f} nats (max possible MI)")

    cond_stats = compute_class_conditional_stats(Z, y)
    direction = np.where(
        cond_stats["contrast_raw"] > 0, LABEL_1_NAME,
        np.where(cond_stats["contrast_raw"] < 0, LABEL_0_NAME, "neutral"),
    )

    active_mask = (
        (cond_stats["activity_rate_all"] >= min_activity_rate)
        & (cond_stats["active_count_all"] >= min_active_count)
        & (cond_stats["var_all_raw"] > 1e-12)
    )
    active_feature_ids = np.where(active_mask)[0]
    if len(active_feature_ids) == 0:
        raise ValueError("No active features after filtering.")
    print(f"  Active features: {len(active_feature_ids)} / {sae_dim} ({100*len(active_feature_ids)/sae_dim:.1f}%)")

    def zero_array():
        return np.zeros(sae_dim, dtype=np.float64)

    mi_score = zero_array()
    nmi_score = zero_array()
    mi_mean_over_k = zero_array()
    mi_std_over_k = zero_array()
    mi_boot_mean = zero_array()
    mi_boot_std = zero_array()
    mi_boot_q05 = zero_array()
    mi_boot_q95 = zero_array()
    mi_debiased = zero_array()
    mi_zscore = zero_array()
    p_value_perm = np.ones(sae_dim, dtype=np.float64)
    null_mean = zero_array()

    Z_stat = apply_transform(Z, transform)
    X_active = Z_stat[:, active_feature_ids]

    print("  Computing single-shot MI at default k...")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi_active = mutual_info_classif(
            X_active, y,
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
    print(f"    Done in {time.time() - t0:.1f}s")

    mi_active = np.nan_to_num(mi_active, nan=0.0, posinf=0.0, neginf=0.0)
    mi_active = np.maximum(mi_active, 0.0)
    mi_score[active_feature_ids] = mi_active
    nmi_score[active_feature_ids] = mi_active / (h_y + 1e-12)

    multi_k_results = compute_mi_multi_k(X_active, y, k_values=multi_k, random_state=random_state)
    mi_mean_over_k[active_feature_ids] = multi_k_results["mi_mean_over_k"]
    mi_std_over_k[active_feature_ids] = multi_k_results["mi_std_over_k"]

    if do_bootstrap:
        boot = compute_mi_with_bootstrap(
            X_active, y,
            n_bootstrap=n_bootstrap,
            n_neighbors=n_neighbors,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        mi_boot_mean[active_feature_ids] = boot["mi_mean"]
        mi_boot_std[active_feature_ids] = boot["mi_std"]
        mi_boot_q05[active_feature_ids] = boot["mi_q05"]
        mi_boot_q95[active_feature_ids] = boot["mi_q95"]
        n_bootstraps_used = boot["n_successful_bootstraps"]
    else:
        n_bootstraps_used = 0

    if do_permutation:
        perm = compute_mi_permutation_pvalue(
            X_active, y, observed_mi=mi_active,
            n_permutations=n_permutations,
            n_neighbors=n_neighbors,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        p_value_perm[active_feature_ids] = perm["p_value_perm"]
        null_mean[active_feature_ids] = perm["null_mean"]
        mi_debiased[active_feature_ids] = perm["mi_debiased"]
        mi_zscore[active_feature_ids] = perm["mi_zscore"]
        q_value_perm = benjamini_hochberg(p_value_perm)
    else:
        q_value_perm = np.ones(sae_dim, dtype=np.float64)

    mi_boot_cv = np.where(
        mi_boot_mean > 1e-12,
        mi_boot_std / (mi_boot_mean + 1e-12),
        np.inf,
    )

    results = pd.DataFrame({
        "layer": layer,
        "feature_id": np.arange(sae_dim),
        "mi_score": mi_score,
        "nmi_score": nmi_score,
        "mi_debiased": mi_debiased,
        "null_mean": null_mean,
        "p_value_perm": p_value_perm,
        "q_value_perm_bh": q_value_perm,
        "mi_zscore": mi_zscore,
        "mi_mean_over_k": mi_mean_over_k,
        "mi_std_over_k": mi_std_over_k,
        "mi_boot_mean": mi_boot_mean,
        "mi_boot_std": mi_boot_std,
        "mi_boot_q05": mi_boot_q05,
        "mi_boot_q95": mi_boot_q95,
        "mi_boot_cv": mi_boot_cv,
        "is_active_feature": active_mask,
        "active_count_all": cond_stats["active_count_all"],
        "activity_rate_all": cond_stats["activity_rate_all"],
        "activity_rate_right": cond_stats["activity_rate_right"],
        "activity_rate_left": cond_stats["activity_rate_left"],
        "mean_right_raw": cond_stats["mean_right_raw"],
        "mean_left_raw": cond_stats["mean_left_raw"],
        "contrast_right_minus_left_raw": cond_stats["contrast_raw"],
        "direction_raw": direction,
        "transform_used_for_mi": transform,
        "n_neighbors_default": n_neighbors,
    })

    for k_col_name, k_values_active in multi_k_results["mi_per_k"].items():
        full = zero_array()
        full[active_feature_ids] = k_values_active
        results[k_col_name] = full

    rank_score = mi_debiased if do_permutation else mi_score
    order = np.argsort(rank_score)[::-1]
    results = results.iloc[order].reset_index(drop=True)
    results["mi_rank"] = np.arange(1, len(results) + 1)
    cols = results.columns.tolist()
    cols.remove("mi_rank")
    results = results[["mi_rank"] + cols]

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
        "h_y_nats": float(h_y),
        "max_mi_score": float(results.iloc[0]["mi_score"]),
        "max_nmi_score": float(results.iloc[0]["nmi_score"]),
        "max_mi_debiased": float(results.iloc[0]["mi_debiased"]),
        "top_feature_id": int(results.iloc[0]["feature_id"]),
        "top_feature_direction": str(results.iloc[0]["direction_raw"]),
        "n_neighbors_default": int(n_neighbors),
        "multi_k": list(multi_k),
        "n_bootstraps": int(n_bootstraps_used),
        "n_permutations": int(n_permutations) if do_permutation else 0,
        "num_features_q_less_0_05": int(np.sum(q_value_perm < 0.05)),
        "num_features_q_less_0_01": int(np.sum(q_value_perm < 0.01)),
        "num_features_q_less_0_001": int(np.sum(q_value_perm < 0.001)),
    }
    return results, summary


def save_plots(results, Z, y, layer, out_dir, h_y, top_n_dist=10):
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)

    plt.figure(figsize=(10, 6))
    plt.hist(results["mi_score"].values, bins=100)
    plt.axvline(h_y, color="red", linestyle="--", label=f"H(Y) = {h_y:.3f}")
    plt.xlabel("MI score (nats)")
    plt.ylabel("Number of features")
    plt.title(f"Layer {layer}: MI distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"layer_{layer}_mi_histogram.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(results["nmi_score"].values, bins=100)
    plt.xlabel("Normalized MI = MI / H(Y)")
    plt.ylabel("Number of features")
    plt.title(f"Layer {layer}: NMI distribution (1.0 = perfect predictor)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"layer_{layer}_nmi_histogram.png"), dpi=150)
    plt.close()

    top50 = results.head(50)
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(top50)), top50["mi_score"].values)
    plt.xlabel("Rank")
    plt.ylabel("MI (nats)")
    plt.title(f"Layer {layer}: Top 50 MI features")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"layer_{layer}_top50_mi.png"), dpi=150)
    plt.close()

    if (results["mi_boot_mean"] > 0).any():
        active_results = results[results["mi_score"] > 0].copy()
        active_results["mi_boot_cv_capped"] = np.minimum(active_results["mi_boot_cv"], 5.0)
        plt.figure(figsize=(10, 6))
        plt.scatter(active_results["mi_score"], active_results["mi_boot_cv_capped"], s=2, alpha=0.4)
        plt.xlabel("MI score")
        plt.ylabel("Bootstrap CV (lower = more reliable)")
        plt.title(f"Layer {layer}: MI score vs reliability")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"layer_{layer}_mi_vs_cv.png"), dpi=150)
        plt.close()

    if (results["null_mean"] > 0).any():
        active_results = results[results["mi_score"] > 0]
        plt.figure(figsize=(10, 6))
        plt.scatter(active_results["mi_score"], active_results["null_mean"], s=2, alpha=0.4)
        plt.plot(
            [0, active_results["mi_score"].max()],
            [0, active_results["mi_score"].max()],
            "r--",
            label="y = x",
        )
        plt.xlabel("Observed MI")
        plt.ylabel("Null MI (estimator bias)")
        plt.title(f"Layer {layer}: MI vs estimator bias")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"layer_{layer}_mi_vs_null.png"), dpi=150)
        plt.close()

    right_mask = y == 1
    left_mask = y == 0
    for _, row in results.head(top_n_dist).iterrows():
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
            f"MI={row['mi_score']:.4f}, NMI={row['nmi_score']:.3f}, "
            f"direction={row['direction_raw']}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"layer_{layer}_feature_{fid}_distribution.png"), dpi=150)
        plt.close()


def run_layer(
    base_dir, out_dir, layer, transform,
    min_activity_rate, min_active_count,
    allowed_splits, top_n,
    n_neighbors, multi_k, n_bootstrap, n_permutations,
    do_bootstrap, do_permutation,
    n_jobs, random_state, make_plots,
):
    input_path, Z, y, metadata, keys = load_layer_file(base_dir, layer)
    Z_use, y_use, metadata_use, row_mask = filter_rows(Z, y, metadata, allowed_splits=allowed_splits)

    print("\n" + "=" * 90)
    print(f"LAYER {layer}")
    print("=" * 90)
    print("Input:", input_path)
    print("Z shape:", Z.shape, " | Used:", Z_use.shape)
    print(f"Right: {int(np.sum(y_use == 1))} | Left: {int(np.sum(y_use == 0))}")
    print("Transform:", transform)

    split_tag = "all_rows" if not allowed_splits else "_".join([str(s).strip().lower() for s in allowed_splits])
    results, summary = compute_complete_mi(
        Z=Z_use, y=y_use, layer=layer,
        transform=transform,
        min_activity_rate=min_activity_rate,
        min_active_count=min_active_count,
        n_neighbors=n_neighbors,
        multi_k=multi_k,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        do_bootstrap=do_bootstrap,
        do_permutation=do_permutation,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    summary["input_path"] = input_path
    summary["rows_used"] = int(Z_use.shape[0])
    summary["row_filter"] = split_tag
    if "split2" in metadata_use:
        summary["split2_counts_used"] = {
            k: int(v) for k, v in pd.Series(as_str_array(metadata_use["split2"])).value_counts().sort_index().to_dict().items()
        }
    if "category" in metadata_use:
        summary["category_counts_used"] = {
            k: int(v) for k, v in pd.Series(as_str_array(metadata_use["category"])).value_counts().sort_index().to_dict().items()
        }
    if "topic" in metadata_use:
        summary["num_topics_used"] = int(len(set(as_str_array(metadata_use["topic"]))))

    layer_out_dir = os.path.join(out_dir, f"layer_{layer}")
    ensure_dir(layer_out_dir)

    suffix = f"{split_tag}_{transform}"
    full_csv = os.path.join(layer_out_dir, f"layer_{layer}_mi_full_{suffix}.csv")
    top_csv = os.path.join(layer_out_dir, f"layer_{layer}_mi_top{top_n}_{suffix}.csv")
    top_npy = os.path.join(layer_out_dir, f"layer_{layer}_mi_top{top_n}_ids_{suffix}.npy")
    summary_json = os.path.join(layer_out_dir, f"layer_{layer}_mi_summary_{suffix}.json")
    scores_npz = os.path.join(layer_out_dir, f"layer_{layer}_mi_scores_{suffix}.npz")

    results.to_csv(full_csv, index=False)
    results.head(top_n).to_csv(top_csv, index=False)
    np.save(top_npy, results.head(top_n)["feature_id"].astype(np.int32).values)
    safe_json_dump(summary, summary_json)

    feature_id_in_order = np.arange(summary["sae_dim"])
    results_by_id = results.sort_values("feature_id")
    np.savez_compressed(
        scores_npz,
        feature_id=feature_id_in_order,
        mi_score=results_by_id["mi_score"].values,
        nmi_score=results_by_id["nmi_score"].values,
        mi_debiased=results_by_id["mi_debiased"].values,
        mi_zscore=results_by_id["mi_zscore"].values,
        p_value_perm=results_by_id["p_value_perm"].values,
        q_value_perm_bh=results_by_id["q_value_perm_bh"].values,
        mi_boot_mean=results_by_id["mi_boot_mean"].values,
        mi_boot_std=results_by_id["mi_boot_std"].values,
        mi_boot_cv=results_by_id["mi_boot_cv"].values,
        mi_mean_over_k=results_by_id["mi_mean_over_k"].values,
        mi_std_over_k=results_by_id["mi_std_over_k"].values,
        contrast_right_minus_left_raw=results_by_id["contrast_right_minus_left_raw"].values,
        activity_rate_all=results_by_id["activity_rate_all"].values,
        is_active_feature=results_by_id["is_active_feature"].values,
    )

    print("\nTop 20 features:")
    print(results.head(20)[
        ["mi_rank", "feature_id", "mi_score", "nmi_score",
         "mi_debiased", "p_value_perm", "q_value_perm_bh",
         "mi_boot_cv", "activity_rate_all", "direction_raw"]
    ].to_string(index=False))

    print(f"\nSaved:\n  {full_csv}\n  {top_csv}\n  {top_npy}\n  {scores_npz}\n  {summary_json}")

    if make_plots:
        save_plots(results=results, Z=Z_use, y=y_use, layer=layer, out_dir=layer_out_dir, h_y=summary["h_y_nats"])
        print("Plots saved under:", os.path.join(layer_out_dir, "plots"))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Complete Mutual Information analysis for political polarity SAE features.")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--transform", choices=["raw", "log1p"], default="log1p")
    parser.add_argument("--min_activity_rate", type=float, default=0.01)
    parser.add_argument("--min_active_count", type=int, default=5)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--top_n", type=int, default=2000)
    parser.add_argument("--n_neighbors", type=int, default=3)
    parser.add_argument("--multi_k", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--n_bootstrap", type=int, default=20)
    parser.add_argument("--n_permutations", type=int, default=100)
    parser.add_argument("--no_bootstrap", action="store_true")
    parser.add_argument("--no_permutation", action="store_true")
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
            n_neighbors=args.n_neighbors,
            multi_k=tuple(args.multi_k),
            n_bootstrap=args.n_bootstrap,
            n_permutations=args.n_permutations,
            do_bootstrap=not args.no_bootstrap,
            do_permutation=not args.no_permutation,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            make_plots=args.make_plots,
        )
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_csv = os.path.join(args.out_dir, f"mi_all_layers_summary_{args.transform}.csv")
    summary_json = os.path.join(args.out_dir, f"mi_all_layers_summary_{args.transform}.json")
    summary_df.to_csv(summary_csv, index=False)
    safe_json_dump(all_summaries, summary_json)

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print("All-layer summary:", summary_csv)


if __name__ == "__main__":
    main()
