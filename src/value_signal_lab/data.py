"""Seeded simulation using entirely synthetic customer records."""

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "first_order_revenue",
    "sessions_before_purchase",
    "product_views_before_purchase",
    "discount_rate",
]
CATEGORICAL_FEATURES = ["channel", "device"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "revenue_90d"
AS_OF = "2026-11-01"


def simulate_customers(n=12000, seed=42, as_of=AS_OF, experiment=False, effect=0.12):
    """Generate cumulative revenue; mask outcomes whose observation window is incomplete.

    Features are measured by the first purchase. Treatment in the separate experiment
    changes subsequent revenue and cost. Its effect is an assumed scenario, not a finding.
    """
    if n < 100:
        raise ValueError("Use pelo menos 100 clientes para a simulação.")
    rng = np.random.default_rng(seed)
    start, end = ("2026-01-01", "2026-05-31") if experiment else ("2024-01-01", "2026-09-30")
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    acquired = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    channel = rng.choice(["Google", "Meta", "TikTok"], n, p=[0.40, 0.36, 0.24])
    device = rng.choice(["mobile", "desktop"], n, p=[0.78, 0.22])
    sessions = 1 + rng.poisson(2.8, n)
    views = sessions + rng.poisson(sessions * 2.2)
    discount = rng.choice([0.0, 0.1, 0.2, 0.3], n, p=[0.42, 0.28, 0.22, 0.08])
    channel_value = np.select([channel == "Google", channel == "Meta"], [0.22, 0.06], -0.17)
    first = rng.lognormal(3.65 + 0.12 * (device == "desktop"), 0.48, n) * (1 - discount)
    quality = np.exp(
        -0.6
        + 0.19 * sessions
        + 0.025 * views
        + channel_value
        - 1.6 * discount
        + rng.normal(0, 0.42, n)
    )
    repeat_count = rng.poisson(np.minimum(quality, 15))
    future = repeat_count * first * rng.lognormal(-0.12, 0.46, n)
    cost = rng.lognormal(
        np.log(np.select([channel == "Google", channel == "Meta"], [39.0, 28.0], 21.0)), 0.30, n
    )
    arm = rng.choice(["control", "value_signal"], n) if experiment else np.repeat("historical", n)
    if experiment:
        if effect <= -1:
            raise ValueError("O efeito deve ser maior que -1.")
        future *= np.where(arm == "value_signal", 1 + effect, 1)
        cost *= np.where(arm == "value_signal", 1.02, 1)
    df = pd.DataFrame(
        {
            "customer_id": [f"{'exp' if experiment else 'hist'}_{i:06d}" for i in range(n)],
            "acquired_at": acquired,
            "channel": channel,
            "device": device,
            "sessions_before_purchase": sessions,
            "product_views_before_purchase": views,
            "discount_rate": discount,
            "first_order_revenue": first.round(2),
            "acquisition_cost": cost.round(2),
            "arm": arm,
        }
    )
    for horizon, fraction in [(30, 0.48), (60, 0.79), (90, 1.0)]:
        df[f"revenue_{horizon}d"] = np.where(
            (pd.Timestamp(as_of) - acquired).days >= horizon,
            np.round(first + future * fraction, 2),
            np.nan,
        )
    return df.sort_values(["acquired_at", "customer_id"]).reset_index(drop=True)


def validate_customers(df, as_of=AS_OF, experiment=False):
    """Reject ambiguous units, invalid amounts and incomplete *eligible* outcomes."""
    required = ["customer_id", "acquired_at", "acquisition_cost", TARGET] + FEATURES
    if experiment:
        required += ["arm"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    df = df.copy()
    df["acquired_at"] = pd.to_datetime(df["acquired_at"], errors="raise")
    if df["customer_id"].isna().any() or df["customer_id"].duplicated().any():
        raise ValueError("customer_id deve ser único e preenchido (uma linha por cliente).")
    if df["acquired_at"].isna().any() or (df["acquired_at"] > pd.Timestamp(as_of)).any():
        raise ValueError("Data de aquisição ausente ou posterior à data de corte.")
    for col in NUMERIC_FEATURES + ["acquisition_cost", TARGET]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        values = df[col].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{col}: valores devem ser finitos e não negativos.")
    if df[FEATURES + ["acquisition_cost"]].isna().any().any():
        raise ValueError("Features e custos não podem ter valores ausentes.")
    if (df["acquisition_cost"] <= 0).any():
        raise ValueError("acquisition_cost deve ser positivo.")
    if not df["discount_rate"].between(0, 1).all():
        raise ValueError("discount_rate deve estar entre 0 e 1.")
    mature = df["acquired_at"] + pd.Timedelta(days=90) <= pd.Timestamp(as_of)
    if df.loc[mature, TARGET].isna().any():
        raise ValueError("Receita de 90 dias ausente para cliente com janela completa.")
    if (df.loc[mature, TARGET] < df.loc[mature, "first_order_revenue"]).any():
        raise ValueError(
            "Receita acumulada não pode ser menor que a primeira compra neste contrato."
        )
    if experiment and (df["arm"].isna().any() or not set(df["arm"]) <= {"control", "value_signal"}):
        raise ValueError("Braços permitidos: control e value_signal.")
    # Never accept available-looking future labels as mature observations.
    df.loc[~mature, TARGET] = np.nan
    return df


def mature_customers(df, as_of=AS_OF):
    return df.loc[
        (df["acquired_at"] + pd.Timedelta(days=90) <= pd.Timestamp(as_of)) & df[TARGET].notna()
    ].copy()


def channel_summary(df):
    summary = df.groupby("channel", observed=True).agg(
        customers=("customer_id", "size"),
        revenue_90d=(TARGET, "sum"),
        spend=("acquisition_cost", "sum"),
        mean_value_90d=(TARGET, "mean"),
    )
    summary["cac"] = summary["spend"] / summary["customers"]
    summary["roas_90d"] = summary["revenue_90d"] / summary["spend"]
    return summary.reset_index()
