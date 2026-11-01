import json

import numpy as np
import pandas as pd
import pytest

from value_signal_lab.cli import run_analysis
from value_signal_lab.data import TARGET, simulate_customers


def test_end_to_end_artifacts_and_finite_metrics(tmp_path):
    result = run_analysis(tmp_path, n=2000, n_boot=100)
    assert (tmp_path / "report.html").exists()
    assert "DADOS SINTÉTICOS" in (tmp_path / "report.html").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == result
    deciles = pd.read_csv(tmp_path / "calibration.csv")
    assert len(deciles) == 10
    assert np.isfinite(list(result["prediction"]["model"].values())).all()
    assert result["prediction"]["model"]["mae"] < result["prediction"]["baseline"]["mae"]


def test_csv_mode_does_not_emit_synthetic_report(tmp_path):
    historical = tmp_path / "historical.csv"
    experiment = tmp_path / "experiment.csv"
    simulate_customers(2000).to_csv(historical, index=False)
    simulate_customers(1000, seed=43, experiment=True).to_csv(experiment, index=False)
    output = tmp_path / "results"
    result = run_analysis(output, n_boot=100, historical_csv=historical, experiment_csv=experiment)
    assert not (output / "report.html").exists()
    assert result["provenance"]["experiment"] == "user_csv"
    assert result["provenance"]["simulation_assumptions"] is None


def test_missing_csv_pair_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="ambos"):
        run_analysis(tmp_path, historical_csv="missing.csv")


def test_test_labels_never_affect_model_fitting():
    from value_signal_lab.model import fit_value_model

    df = simulate_customers(2000).dropna(subset=[TARGET])
    _, first, _ = fit_value_model(df)
    changed = df.copy()
    test = changed.acquired_at.between("2025-10-01", "2026-07-31")
    changed.loc[test, TARGET] *= 10
    _, second, _ = fit_value_model(changed)
    np.testing.assert_array_equal(first.predicted_value_90d, second.predicted_value_90d)
