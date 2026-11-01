"""Run one deterministic analysis and write auditable artifacts."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd

from .data import (
    AS_OF,
    channel_summary,
    mature_customers,
    simulate_customers,
    validate_customers,
)
from .experiment import analyze_experiment, plan_sample_size
from .model import fit_value_model, temporal_split
from .report import render_report


def run_analysis(
    output, n=12000, seed=42, n_boot=2000, as_of=AS_OF, historical_csv=None, experiment_csv=None
):
    output = Path(output)
    if bool(historical_csv) != bool(experiment_csv):
        raise ValueError("Forneça ambos os CSVs para não misturar fontes reais e sintéticas.")
    if historical_csv and (output / "report.html").exists():
        raise ValueError(
            "A pasta contém um HTML sintético anterior. Use uma pasta nova para os CSVs próprios."
        )
    if (historical_csv or experiment_csv) and output.resolve() == (
        Path(__file__).resolve().parents[2] / "reports" / "demo"
    ):
        raise ValueError(
            "Para CSV próprio, use --output reports/local para não sobrescrever a demo pública."
        )
    output.mkdir(parents=True, exist_ok=True)
    historical = (
        pd.read_csv(historical_csv) if historical_csv else simulate_customers(n, seed, as_of)
    )
    historical = validate_customers(historical, as_of)
    experiment = (
        pd.read_csv(experiment_csv)
        if experiment_csv
        else simulate_customers(max(n // 2, 100), seed + 1, as_of, experiment=True)
    )
    experiment = validate_customers(experiment, as_of, experiment=True)
    mature = mature_customers(historical, as_of)
    channels = channel_summary(mature)
    metrics, scored, deciles = fit_value_model(mature, seed)
    ab, lifts = analyze_experiment(experiment, n_boot, seed)
    plan = plan_sample_size(temporal_split(mature)["train"])
    summary = {
        "provenance": {
            "historical": "user_csv" if historical_csv else "synthetic",
            "experiment": "user_csv" if experiment_csv else "synthetic",
            "seed": seed,
            "as_of": as_of,
            "simulation_assumptions": {
                "future_revenue_treatment_multiplier": 1.12,
                "treatment_cost_multiplier": 1.02,
            }
            if not experiment_csv
            else None,
        },
        "historical": {
            "total_customers": len(historical),
            "mature_customers": len(mature),
            "excluded_immature": len(historical) - len(mature),
        },
        "prediction": metrics,
        "experiment": ab,
        "planning": plan,
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "scipy", "scikit-learn", "plotly"]
            },
        },
    }
    # Custom records remain local. The bundled public demo is always synthetic.
    if not historical_csv and not experiment_csv:
        render_report(
            historical,
            mature,
            channels,
            metrics,
            scored,
            deciles,
            ab,
            lifts,
            plan,
            output / "report.html",
            as_of,
        )
    channels.to_csv(output / "channels.csv", index=False)
    deciles.to_csv(output / "calibration.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    if not historical_csv and not experiment_csv:
        ci = ab["ci95"]
        (output / "README.md").write_text(
            "# Resultados da demonstração\n\n"
            "**Dados inteiramente sintéticos. Não representam resultados comerciais reais.**\n\n"
            f"- Seed: {seed}; data de corte: {as_of}.\n"
            f"- Clientes históricos: {len(historical):,}; maduros: {len(mature):,}.\n"
            f"- MAE: R$ {metrics['model']['mae']:.2f}; baseline: R$ {metrics['baseline']['mae']:.2f}.\n"
            f"- R² no teste temporal: {metrics['model']['r2']:.3f}.\n"
            f"- Lift relativo de ROAS: {ab['relative_lift']:.1%}; IC 95%: [{ci[0]:.1%}, {ci[1]:.1%}].\n"
            f"- Decisão: {ab['decision']}\n"
            f"- Planejamento: {plan['customers_per_arm']:,} clientes/braço para MDE de 5%.\n\n"
            "Abra `report.html` após baixar o repositório para explorar os gráficos offline. "
            "`summary.json`, `channels.csv` e `calibration.csv` contêm os resultados auditáveis.\n",
            encoding="utf-8",
        )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Análise de valor do cliente e experimento de ROAS."
    )
    parser.add_argument("--output", default="reports/local")
    parser.add_argument("--customers", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--as-of", default=AS_OF)
    parser.add_argument("--historical-csv", type=Path)
    parser.add_argument("--experiment-csv", type=Path)
    args = parser.parse_args()
    if bool(args.historical_csv) != bool(args.experiment_csv):
        parser.error(
            "Forneça os dois CSVs para não misturar dados próprios com experimento simulado."
        )
    try:
        summary = run_analysis(
            args.output,
            args.customers,
            args.seed,
            args.bootstrap,
            args.as_of,
            args.historical_csv,
            args.experiment_csv,
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"Erro: {error}\n")
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "decision": summary["experiment"]["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
