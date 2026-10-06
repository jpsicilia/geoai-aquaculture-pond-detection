"""
features.py — Domain-invariant feature engineering for aquaculture pond detection.

The core problem of this challenge is a TEMPORAL DOMAIN SHIFT: the model is
trained on one acquisition period and evaluated on a different one. An
adversarial classifier can tell train from test with AUC ~1.0, so features that
look discriminative on the training domain often fail to transfer.

This module turns each pixel's 12-month x 12-band satellite time series into a
compact set of descriptors chosen to be ROBUST to that shift. Two design
decisions matter most:

  1. Sentinel values (-9999, i.e. "no observation") become NaN and every
     temporal statistic is computed NaN-aware, so a pixel seen in 5 months is
     comparable to one seen in 12.

  2. PERCENTILE descriptors (median, p25, p75) are preferred over the mean.
     The test set is missing *different* months than training, so a mean over a
     handful of randomly observed months is noisy; a percentile is far more
     stable to *which* months happen to be present. This single change drove the
     largest leaderboard gain (0.881 -> 0.907).

All indices are standard water / vegetation / red-edge formulas; no external
data is used, in line with the competition rules.
"""

import warnings

import numpy as np
import pandas as pd

# A pixel may have zero observed months for a given index; the NaN-aware
# reducers then warn on an all-NaN slice and correctly return NaN (later
# handled by LightGBM's native missing-value support). The warning is noise.
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
# Sentinel-2 optical + Sentinel-1 SAR bands, each sampled monthly (_01 .. _12).
_EPS = 1e-6


def _monthly_indices(d: pd.DataFrame) -> dict:
    """Compute spectral/radar indices for every month, returning a dict of
    {index_name: array of shape (n_samples, 12)}. NaNs mark missing months."""
    stacks: dict = {}
    for i in range(1, 13):
        m = f"{i:02d}"
        g, n, r = d[f"green_{m}"], d[f"nir_{m}"], d[f"red_{m}"]
        s1, s2, b = d[f"swir1_{m}"], d[f"swir2_{m}"], d[f"blue_{m}"]
        vv, vh = d[f"VV_{m}"], d[f"VH_{m}"]
        re1, re2, re3 = d[f"re1_{m}"], d[f"re2_{m}"], d[f"re3_{m}"]

        # Water indices — a pond is permanent, managed water.
        stacks.setdefault("NDWI", []).append((g - n) / (g + n + _EPS))
        stacks.setdefault("MNDWI", []).append((g - s1) / (g + s1 + _EPS))
        stacks.setdefault("AWEI", []).append(
            (b + 2.5 * g - 1.5 * (n + s1) - 0.25 * s2) / 10000.0
        )
        stacks.setdefault("NDWIre", []).append((g - re2) / (g + re2 + _EPS))
        # Vegetation — distinguishes ponds from surrounding crops/wetland.
        stacks.setdefault("NDVI", []).append((n - r) / (n + r + _EPS))
        # Red-edge / chlorophyll — algae & feed give aquaculture water a
        # distinctive red-edge response vs. natural water bodies.
        stacks.setdefault("NDRE", []).append((n - re1) / (n + re1 + _EPS))
        stacks.setdefault("CIRE", []).append((re3 / (re1 + _EPS)) - 1)
        # Turbidity proxy — suspended matter from feed/algae.
        stacks.setdefault("turb", []).append((r - g) / (r + g + _EPS))
        # SAR — VV and VH are in dB, so their "ratio" is the difference.
        stacks.setdefault("VVmVH", []).append(vv - vh)
        stacks.setdefault("VV", []).append(vv)
        stacks.setdefault("VH", []).append(vh)
    return {k: np.column_stack(v) for k, v in stacks.items()}


def build_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Build the domain-invariant feature table for a raw challenge dataframe.

    Parameters
    ----------
    df : raw Train/Test dataframe (must contain the 144 band_month columns).
    feature_cols : the 144 raw band_month column names (used to mask -9999).

    Returns
    -------
    DataFrame of engineered features, one row per input pixel.
    """
    d = df.copy()
    # "No observation" sentinel -> NaN so statistics ignore it.
    d[feature_cols] = d[feature_cols].where(d[feature_cols] != -9999, np.nan)

    out = pd.DataFrame(index=d.index)
    st = _monthly_indices(d)

    # --- Per-index temporal descriptors (all NaN-aware) ---
    for name, series in st.items():
        out[f"{name}_mean"] = np.nanmean(series, axis=1)
        out[f"{name}_min"] = np.nanmin(series, axis=1)
        out[f"{name}_max"] = np.nanmax(series, axis=1)
        out[f"{name}_std"] = np.nanstd(series, axis=1)
        # Percentiles: robust to WHICH months are missing — the key to transfer.
        out[f"{name}_p50"] = np.nanmedian(series, axis=1)
        out[f"{name}_p25"] = np.nanpercentile(series, 25, axis=1)
        out[f"{name}_p75"] = np.nanpercentile(series, 75, axis=1)

    # --- Water-persistence features (physical signature of a managed pond) ---
    ndwi, mndwi, awei = st["NDWI"], st["MNDWI"], st["AWEI"]
    out["freq_ndwi"] = np.nanmean(ndwi > 0, axis=1)       # fraction of wet months
    out["freq_mndwi"] = np.nanmean(mndwi > 0, axis=1)
    out["freq_awei"] = np.nanmean(awei > 0, axis=1)
    out["water_persist"] = np.nanmean((ndwi > 0) & (mndwi > 0), axis=1)
    out["water_stability"] = 1.0 / (np.nanstd(ndwi, axis=1) + _EPS)

    # --- Interaction features: permanent water that is also strongly water ---
    out["permwater_x_max"] = out["freq_ndwi"] * out["NDWI_max"]
    out["algae_in_water"] = out["freq_ndwi"] * out["NDRE_mean"]
    out["cv_water"] = np.nanstd(ndwi, axis=1) / (
        np.abs(np.nanmean(ndwi, axis=1)) + _EPS
    )
    return out


def rank_transferability(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y: np.ndarray
) -> pd.DataFrame:
    """Score every feature on two axes and rank by net transferability.

    label_auc  : how well the feature separates ponds (on the training domain).
    domain_auc : how well it separates train from test (0.5 = invariant,
                 high = the feature leaks the domain and will not transfer).

    The selection score rewards discrimination and penalises domain leakage
    twice as hard, so only features that are BOTH predictive and invariant rank
    at the top. This is a feature-level form of adversarial validation.
    """
    from sklearn.metrics import roc_auc_score

    dom = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]
    rows = []
    for c in X_train.columns:
        a = X_train[c].fillna(X_train[c].median())
        label_auc = (
            max(roc_auc_score(y, a), 1 - roc_auc_score(y, a))
            if a.nunique() > 1
            else 0.5
        )
        dv = np.r_[a, X_test[c].fillna(X_train[c].median())]
        domain_auc = (
            max(roc_auc_score(dom, dv), 1 - roc_auc_score(dom, dv))
            if len(np.unique(dv)) > 1
            else 0.5
        )
        rows.append((c, label_auc, domain_auc))

    rank = pd.DataFrame(rows, columns=["feature", "label_auc", "domain_auc"])
    rank["score"] = rank["label_auc"] - 2 * (rank["domain_auc"] - 0.5)
    return rank.sort_values("score", ascending=False).reset_index(drop=True)
