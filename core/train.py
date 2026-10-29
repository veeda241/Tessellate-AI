import json
import os
import platform
import sys
from datetime import datetime, timezone

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from core.config import DATA_DIR, MODELS_DIR

# Windows cp1252 consoles cannot render guardrail log emoji — degrade safely instead of crashing
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FEATURES = [
    "track_id",
    "laps_remaining",
    "current_position",
    "tire_age",
    "tire_compound",
    "gap_ahead",
    "gap_behind",
    "driver_pace_index",
    "will_pit",
]
CAT_FEATURES = ["track_id", "tire_compound"]

CV_PARAMS = dict(iterations=250, learning_rate=0.05, depth=6, loss_function="RMSE", random_seed=42)
FINAL_PARAMS = dict(iterations=400, learning_rate=0.05, depth=6, loss_function="RMSE", random_seed=42, verbose=100)


def _group_kfold_indices(groups, n_splits=3, seed=42):
    """Split by whole circuits so each fold is validated on unseen track_ids.

    Implemented with numpy to avoid a scikit-learn dependency; deterministic
    via a seeded shuffle of the unique group labels.
    """
    unique = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    fold_of_group = {g: i % n_splits for i, g in enumerate(unique)}
    fold_of_row = np.array([fold_of_group[g] for g in groups])
    for k in range(n_splits):
        val_idx = np.where(fold_of_row == k)[0]
        train_idx = np.where(fold_of_row != k)[0]
        yield train_idx, val_idx


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return rmse, mae


def train_with_grouped_cv():
    data_path = DATA_DIR / "historical_sc_features.csv"
    if not data_path.exists():
        raise FileNotFoundError("Missing feature logs matrix. Execute core/pipeline.py first.")

    df = pd.read_csv(data_path)
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str)
    df = df.dropna(subset=FEATURES + ["final_position"]).reset_index(drop=True)

    X = df[FEATURES]
    y = df["final_position"]
    groups = df["track_id"]

    print(f"📊 Dataset: {len(df)} rows across {groups.nunique()} distinct circuits")

    # ─── HONEST GROUPED CROSS-VALIDATION (unseen-circuit generalization) ───
    print("🤖 Executing GroupKFold validation across distinct circuit profiles...")
    cv_rmse, cv_mae, cv_baseline_mae = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(_group_kfold_indices(groups), start=1):
        train_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=CAT_FEATURES)
        fold_model = CatBoostRegressor(**CV_PARAMS)
        fold_model.fit(train_pool, silent=True)

        preds = fold_model.predict(X.iloc[va_idx])
        rmse, mae = _metrics(y.iloc[va_idx], preds)
        # Naive baseline: predict the car's current position
        base_mae, _ = _metrics(y.iloc[va_idx], df["current_position"].iloc[va_idx])
        cv_rmse.append(rmse)
        cv_mae.append(mae)
        cv_baseline_mae.append(base_mae)
        held = sorted(groups.iloc[va_idx].unique())
        print(f"  Fold {fold}: RMSE={rmse:.3f} MAE={mae:.3f} (baseline MAE={base_mae:.3f}) | unseen: {held}")

    final_rmse = float(np.mean(cv_rmse))
    final_mae = float(np.mean(cv_mae))
    baseline_mae = float(np.mean(cv_baseline_mae))

    # Train the production model on the full footprint
    print("🏎️ Fitting production model on full dataset...")
    final_pool = Pool(X, y, cat_features=CAT_FEATURES)
    production_model = CatBoostRegressor(**FINAL_PARAMS)
    production_model.fit(final_pool)

    native_out = MODELS_DIR / "strategy_catboost.cbm"
    production_model.save_model(str(native_out), format="cbm")
    print(f"📦 Production C++ layer binary compiled to {native_out}")

    importance = {
        f: round(float(i), 3)
        for f, i in zip(FEATURES, production_model.get_feature_importance())
    }
    metadata = {
        "features": FEATURES,
        "cat_features": CAT_FEATURES,
        "params": FINAL_PARAMS,
        "cv_scheme": "GroupKFold(3) by track_id (unseen-circuit)",
        "holdout_rmse_positions": round(final_rmse, 3),
        "holdout_mae_positions": round(final_mae, 3),
        "baseline_mae_positions": round(baseline_mae, 3),
        "holdout_rmse": round(final_rmse, 3),
        "holdout_mae": round(final_mae, 3),
        "total_training_rows": int(len(df)),
        "n_circuits": int(groups.nunique()),
        "feature_importance": importance,
        "versions": {"catboost": catboost.__version__, "fastf1": fastf1_version(), "python": platform.python_version()},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "serving": "catboost-native",
    }
    metadata_out = MODELS_DIR / "model_metadata.json"
    with open(metadata_out, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"🗂️ Model metadata written to {metadata_out}")

    # Isolated ONNX Branch: failure here never halts server deployment
    try:
        onnx_out = MODELS_DIR / "strategy_catboost.onnx"
        production_model.save_model(
            str(onnx_out), format="onnx", export_parameters={"onnx_domain": "ai.catboost"}
        )
        print(f"✨ ONNX graph structural export finalized: {onnx_out}")
    except Exception as onnx_drift_err:
        if "onnx_out" in locals() and os.path.exists(onnx_out):
            os.remove(onnx_out)
        print(
            f"⚠️ ONNX translation skipped due to package API drift: {onnx_drift_err}. "
            "Continuing with native serving layer."
        )

    print("\n🏁 Production Engine RETRAINED.")
    print(f"🔹 Grouped CV RMSE: {final_rmse:.3f} positions")
    print(f"🔹 Grouped CV MAE : {final_mae:.3f} positions (baseline {baseline_mae:.3f})")


def fastf1_version():
    try:
        import fastf1

        return fastf1.__version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    train_with_grouped_cv()
