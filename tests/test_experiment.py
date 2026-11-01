import numpy as np
import pandas as pd
import pytest

from value_signal_lab.data import TARGET, simulate_customers
from value_signal_lab.experiment import (
    analyze_experiment,
    bootstrap_roas_lift,
    decision,
    plan_sample_size,
    roas,
)


def test_roas_is_ratio_of_sums_and_bootstrap_keeps_pairs():
    control = np.tile([[10.0, 1.0], [20.0, 100.0]], (100, 1))
    assert roas(control) == pytest.approx(30 / 101)
    assert roas(control) != pytest.approx(np.mean(control[:, 0] / control[:, 1]))
    # Revenue is exactly proportional to cost: resampling paired rows must preserve ratio.
    constant_control = np.column_stack([np.arange(1, 201) * 2, np.arange(1, 201)])
    constant_treatment = np.column_stack([np.arange(1, 201) * 3, np.arange(1, 201)])
    result, lifts = bootstrap_roas_lift(constant_control, constant_treatment, 100)
    assert result["relative_lift"] == pytest.approx(0.5)
    np.testing.assert_allclose(lifts, 0.5)


def test_null_and_negative_effect_are_not_reported_as_wins():
    rng = np.random.default_rng(7)
    control = np.column_stack([rng.lognormal(4, 0.4, 400), rng.uniform(10, 30, 400)])
    null, _ = bootstrap_roas_lift(control, control.copy(), 300)
    assert null["ci95"][0] < 0 < null["ci95"][1]
    negative = control.copy()
    negative[:, 0] *= 0.5
    result, _ = bootstrap_roas_lift(control, negative, 300)
    assert result["ci95"][1] < 0
    assert "perda" in decision(result["ci95"], 0.5)


def test_srm_uses_all_assignments_not_only_mature_customers():
    df = simulate_customers(1000, experiment=True)
    df.loc[df.arm == "control", TARGET] = np.nan
    # Keep only 40 mature controls, but every assignment still enters SRM.
    indices = df.index[df.arm == "control"][:40]
    df.loc[indices, TARGET] = np.linspace(90.0, 110.0, len(indices))
    result, _ = analyze_experiment(df, 100)
    assert sum(result["assigned"].values()) == 1000
    assert result["mature_counts"]["control"] == 40
    assert result["excluded_immature"] > 0


def test_practical_threshold_and_srm_gate():
    assert "Investigar" in decision([0.2, 0.4], 0.001)
    assert "Inconclusivo" in decision([0.01, 0.10], 0.5)
    assert "acima" in decision([0.06, 0.15], 0.5)


def test_sample_size_scales_with_inverse_squared_mde():
    df = simulate_customers(2000).dropna(subset=[TARGET])
    small = plan_sample_size(df, minimum_lift=0.05)["customers_per_arm"]
    big = plan_sample_size(df, minimum_lift=0.10)["customers_per_arm"]
    assert small == pytest.approx(4 * big, abs=4)
    with pytest.raises(ValueError):
        plan_sample_size(pd.DataFrame({TARGET: [0] * 40, "acquisition_cost": [10] * 40}))
