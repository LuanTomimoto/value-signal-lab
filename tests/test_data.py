import numpy as np
import pandas as pd
import pytest

from value_signal_lab.data import (
    FEATURES,
    TARGET,
    mature_customers,
    simulate_customers,
    validate_customers,
)
from value_signal_lab.model import temporal_split


def test_reproducible_simulation_and_censoring():
    first = simulate_customers(2000)
    pd.testing.assert_frame_equal(first, simulate_customers(2000))
    immature = first["acquired_at"] > pd.Timestamp("2026-08-03")
    assert first.loc[immature, TARGET].isna().all()
    assert len(mature_customers(first)) < len(first)
    assert (
        first.dropna(subset=[TARGET])[TARGET]
        >= first.dropna(subset=[TARGET])["first_order_revenue"]
    ).all()


def test_temporal_split_has_mature_labels_at_next_stage_start():
    blocks = temporal_split(mature_customers(simulate_customers(3000)))
    train, calibration, test = [blocks[k] for k in ["train", "calibration", "test"]]
    assert train.acquired_at.max() + pd.Timedelta(days=90) < calibration.acquired_at.min()
    assert calibration.acquired_at.max() + pd.Timedelta(days=90) < test.acquired_at.min()
    assert set(train.customer_id).isdisjoint(set(test.customer_id) | set(calibration.customer_id))
    assert set(FEATURES).isdisjoint({TARGET, "revenue_30d", "revenue_60d", "arm", "customer_id"})


@pytest.mark.parametrize(
    "problem", ["duplicate", "negative", "nan_feature", "infinite", "missing_mature"]
)
def test_data_contract_rejects_bad_records(problem):
    df = simulate_customers(1000)
    if problem == "duplicate":
        df.loc[1, "customer_id"] = df.loc[0, "customer_id"]
    elif problem == "negative":
        df.loc[0, "acquisition_cost"] = -1
    elif problem == "nan_feature":
        df.loc[0, "discount_rate"] = np.nan
    elif problem == "infinite":
        df.loc[0, "first_order_revenue"] = np.inf
    else:
        df.loc[0, TARGET] = np.nan
    with pytest.raises(ValueError):
        validate_customers(df)


def test_future_label_is_masked_even_when_csv_contains_it():
    df = simulate_customers(1000)
    immature = df["acquired_at"] > pd.Timestamp("2026-08-03")
    df.loc[immature, TARGET] = 99999
    checked = validate_customers(df)
    assert checked.loc[immature, TARGET].isna().all()
