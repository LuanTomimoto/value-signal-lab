"""Chronological holdout with 90-day label embargoes and untouched test set."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


def temporal_split(df):
    blocks = {
        "train": ("2024-01-01", "2024-12-31"),
        "calibration": ("2025-04-01", "2025-06-30"),
        "test": ("2025-10-01", "2026-07-31"),
    }
    result = {
        name: df.loc[df["acquired_at"].between(start, end) & df[TARGET].notna()].copy()
        for name, (start, end) in blocks.items()
    }
    if any(len(block) < 100 for block in result.values()):
        raise ValueError(
            "Cada bloco temporal precisa de ao menos 100 clientes maduros; veja docs/data-contract.md."
        )
    return result


def evaluate(y, prediction):
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
        "aggregate_bias_pct": float(100 * (np.sum(prediction) / np.sum(y) - 1)),
    }


def fit_value_model(df, seed=42):
    blocks = temporal_split(df)
    train, calibration, test = [blocks[k] for k in ["train", "calibration", "test"]]
    if any(block[TARGET].sum() <= 0 for block in blocks.values()):
        raise ValueError("Cada bloco precisa ter receita total positiva.")
    if train["first_order_revenue"].sum() <= 0:
        raise ValueError(
            "O treino precisa de primeira compra com receita positiva para a baseline."
        )
    preprocessing = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = make_pipeline(
        preprocessing,
        HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=180,
            learning_rate=0.06,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=3.0,
            early_stopping=False,
            random_state=seed,
        ),
    )
    model.fit(train[FEATURES], train[TARGET])
    raw_calibration = model.predict(calibration[FEATURES])
    factor = float(calibration[TARGET].sum() / raw_calibration.sum())
    prediction = model.predict(test[FEATURES]) * factor
    baseline_multiplier = float(train[TARGET].sum() / train["first_order_revenue"].sum())
    baseline = test["first_order_revenue"].to_numpy() * baseline_multiplier
    scored = test.copy()
    scored["predicted_value_90d"] = prediction
    # Rank-based bins handle tied predictions; ties are ordered by stable input row order.
    scored["decile"] = (
        pd.qcut(scored["predicted_value_90d"].rank(method="first"), 10, labels=False) + 1
    )
    deciles = (
        scored.groupby("decile", observed=True)
        .agg(
            predicted=("predicted_value_90d", "mean"),
            observed=(TARGET, "mean"),
            customers=("customer_id", "size"),
        )
        .reset_index()
    )
    metrics = {
        "model": evaluate(test[TARGET], prediction),
        "baseline": evaluate(test[TARGET], baseline),
        "calibration_factor": factor,
        "baseline_multiplier": baseline_multiplier,
        "top_decile_lift": float(deciles.iloc[-1]["observed"] / test[TARGET].mean()),
        "split_sizes": {k: len(v) for k, v in blocks.items()},
        "split_dates": {
            k: [str(v["acquired_at"].min().date()), str(v["acquired_at"].max().date())]
            for k, v in blocks.items()
        },
    }
    return metrics, scored, deciles
