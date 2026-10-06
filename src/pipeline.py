"""
pipeline.py — Reproducible end-to-end pipeline (public LB 0.907, final 0.9109).

Run:
    python pipeline.py
with Train.csv, Test.csv and SampleSubmission.csv in the working directory.

Pipeline:
    1. Build domain-invariant features (see features.py).
    2. Rank features by transferability and keep the top 18.
    3. Train a regularised LightGBM with missing-data augmentation
       (train copies are masked to mimic the test set's monthly gaps).
    4. Optional: one round of confident self-training (test-time adaptation),
       which took the score from 0.907 to 0.9109.
    5. Write the submission using the fixed 0.5 threshold required by the rules.

Everything is seeded; re-running reproduces the leaderboard score exactly.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from features import build_features, rank_transferability

SEED = 42
N_FEATURES = 18
SEEDS = [42, 7, 2024]          # ensemble over 3 seeds x 5 folds
SELF_TRAINING = True           # set False for the plain 0.907 model

LGBM_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, max_depth=4, num_leaves=15,
    min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
    class_weight="balanced", reg_alpha=2.0, reg_lambda=2.0,
    n_jobs=-1, verbose=-1,
)


def mask_like_test(df, raw_cols, test_miss, rng):
    """Inject missing months into a training copy at the test set's monthly
    rates, so the model learns to cope with partial observations."""
    d = df.copy()
    for m in range(1, 13):
        cols = [c for c in raw_cols if c.endswith(f"_{m:02d}")]
        drop = rng.random(len(d)) < test_miss[m]
        d.loc[d.index[drop], cols] = -9999
    return d


def train_and_predict(tr, te, cols, raw_cols, y, test_miss,
                      extra_X=None, extra_y=None, extra_w=None):
    """3-seed x 5-fold LightGBM with missing-data augmentation.
    Optionally appends pseudo-labelled test rows (self-training)."""
    X_test = build_features(te, raw_cols)[cols]
    pred = np.zeros(len(te))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for seed in SEEDS:
        for fold, (ti, _) in enumerate(skf.split(tr, y)):
            rng = np.random.default_rng(seed * 1000 + fold)
            # clean fold + two test-like masked copies
            a = build_features(mask_like_test(tr.iloc[ti], raw_cols, test_miss, rng), raw_cols)[cols]
            b = build_features(mask_like_test(tr.iloc[ti], raw_cols, test_miss, rng), raw_cols)[cols]
            base = build_features(tr.iloc[ti], raw_cols)[cols]
            X = pd.concat([base, a, b]); yy = np.concatenate([y[ti]] * 3)
            w = np.ones(len(yy))
            if extra_X is not None:
                X = pd.concat([X, extra_X]); yy = np.concatenate([yy, extra_y])
                w = np.concatenate([w, extra_w])
            model = LGBMClassifier(**LGBM_PARAMS, random_state=seed).fit(X, yy, sample_weight=w)
            pred += model.predict_proba(X_test)[:, 1]
    return pred / (len(SEEDS) * 5)


def self_train(tr, te, cols, raw_cols, y, test_miss, base_pred,
               threshold=0.85, iters=4, weight_scale=0.5):
    """Confident self-training (a tree-friendly form of test-time adaptation):
    add the most confident test predictions as pseudo-labels, weighted by
    confidence, and retrain. Took the score from 0.907 to 0.9109."""
    X_test = build_features(te, raw_cols)[cols]
    p = base_pred.copy()
    for _ in range(iters):
        confident = np.where((p >= threshold) | (p <= 1 - threshold))[0]
        pseudo_y = (p[confident] >= 0.5).astype(int)
        pseudo_w = np.abs(p[confident] - 0.5) * 2 * weight_scale
        p = train_and_predict(tr, te, cols, raw_cols, y, test_miss,
                              X_test.iloc[confident], pseudo_y, pseudo_w)
    return p


def main():
    tr = pd.read_csv("Train.csv")
    te = pd.read_csv("Test.csv")
    ss = pd.read_csv("SampleSubmission.csv")
    y = tr["label"].values
    raw_cols = [c for c in te.columns if c != "ID"]

    # Monthly missing-data rate in the test set (drives the augmentation).
    test_miss = {
        m: (te[[c for c in raw_cols if c.endswith(f"_{m:02d}")]] == -9999).values.mean()
        for m in range(1, 13)
    }

    # 1-2. Features + transferability-ranked selection.
    X_tr = build_features(tr, raw_cols)
    X_te = build_features(te, raw_cols)
    rank = rank_transferability(X_tr, X_te, y)
    cols = rank["feature"].tolist()[:N_FEATURES]
    print(f"Selected {N_FEATURES} most transferable features:\n{cols}")

    # 3. Base model.
    pred = train_and_predict(tr, te, cols, raw_cols, y, test_miss)
    print(f"Base model — predicted pond rate: {(pred >= 0.5).mean():.1%}")

    # 4. Optional self-training.
    if SELF_TRAINING:
        pred = self_train(tr, te, cols, raw_cols, y, test_miss, pred)
        print(f"After self-training — pond rate: {(pred >= 0.5).mean():.1%}")

    # 5. Submission with the mandatory 0.5 threshold.
    out = pd.DataFrame({"ID": te["ID"], "TargetRAUC": pred,
                        "TargetF1": (pred >= 0.5).astype(int)})
    sub = ss[["ID"]].merge(out, on="ID", how="left")[["ID", "TargetF1", "TargetRAUC"]]
    sub.to_csv("submission.csv", index=False)
    print(f"Wrote submission.csv ({len(sub)} rows)")


if __name__ == "__main__":
    main()
