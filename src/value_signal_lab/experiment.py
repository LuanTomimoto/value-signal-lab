"""Customer-paired bootstrap for ratios and an explicit fixed-horizon decision rule."""

import numpy as np
from scipy import stats

from .data import TARGET


def roas(array):
    """Ratio of sums, never the average of individual revenue/cost ratios."""
    return float(array[:, 0].sum() / array[:, 1].sum())


def bootstrap_roas_lift(control, treatment, n_boot=2000, seed=42):
    if n_boot < 100:
        raise ValueError("Use pelo menos 100 reamostragens.")
    for arm in [control, treatment]:
        if (
            arm.ndim != 2
            or arm.shape[1] != 2
            or len(arm) < 30
            or not np.isfinite(arm).all()
            or (arm[:, 0] < 0).any()
            or (arm[:, 1] <= 0).any()
        ):
            raise ValueError("Cada braço precisa de >=30 pares finitos [receita >=0, custo >0].")
    if control[:, 0].sum() <= 0:
        raise ValueError("Receita do controle deve ser positiva para calcular lift relativo.")
    rng = np.random.default_rng(seed)
    lifts = np.empty(n_boot)
    for i in range(n_boot):
        c = control[rng.integers(0, len(control), len(control))]
        t = treatment[rng.integers(0, len(treatment), len(treatment))]
        c_roas = roas(c)
        if c_roas <= 0:
            raise ValueError("Bootstrap degenerado: receita do controle nula em uma reamostra.")
        lifts[i] = roas(t) / c_roas - 1
    return {
        "control_roas": roas(control),
        "treatment_roas": roas(treatment),
        "relative_lift": roas(treatment) / roas(control) - 1,
        "ci95": [float(x) for x in np.quantile(lifts, [0.025, 0.975])],
        "bootstrap_replicates": n_boot,
    }, lifts


def decision(ci, srm_pvalue, minimum_lift=0.05):
    if srm_pvalue < 0.01:
        return "Investigar desequilíbrio de alocação antes de interpretar o efeito."
    if ci[0] > minimum_lift:
        return "Evidência de ganho acima do limiar prático na amostra analisada."
    if ci[1] < 0:
        return "Evidência de perda na amostra analisada."
    return "Inconclusivo para adoção: o intervalo não sustenta ganho acima do limiar prático."


def analyze_experiment(df, n_boot=2000, seed=42, minimum_lift=0.05):
    # Check allocation on ALL assigned users, before maturation filtering.
    counts = df["arm"].value_counts().reindex(["control", "value_signal"], fill_value=0)
    srm = float(stats.chisquare(counts.to_numpy(), f_exp=np.repeat(len(df) / 2, 2)).pvalue)
    mature = df.loc[df[TARGET].notna()]
    control = mature.loc[mature["arm"] == "control", [TARGET, "acquisition_cost"]].to_numpy()
    treatment = mature.loc[mature["arm"] == "value_signal", [TARGET, "acquisition_cost"]].to_numpy()
    result, lifts = bootstrap_roas_lift(control, treatment, n_boot, seed)
    welch = stats.ttest_ind(treatment[:, 0], control[:, 0], equal_var=False)
    result.update(
        {
            "assigned": {k: int(v) for k, v in counts.items()},
            "mature_counts": {"control": len(control), "value_signal": len(treatment)},
            "excluded_immature": len(df) - len(mature),
            "srm_pvalue": srm,
            "secondary_welch_revenue_pvalue": float(welch.pvalue),
            "minimum_practical_lift": minimum_lift,
            "decision": decision(result["ci95"], srm, minimum_lift),
        }
    )
    return result, lifts


def plan_sample_size(historical, minimum_lift=0.05, power=0.80, alpha=0.05):
    """Normal approximation for a ratio metric using baseline influence residuals.

    Equal-size, independent arms; same variance/cost distribution under the alternative.
    Planning input is historical data, never the realized treatment effect.
    """
    if not 0 < minimum_lift < 1 or not 0 < power < 1 or not 0 < alpha < 1:
        raise ValueError("MDE, poder e alpha devem estar entre 0 e 1.")
    r = historical[TARGET].to_numpy()
    c = historical["acquisition_cost"].to_numpy()
    if (
        len(r) < 30
        or not np.isfinite(r).all()
        or not np.isfinite(c).all()
        or r.mean() <= 0
        or (c <= 0).any()
    ):
        raise ValueError(
            "Planejamento exige >=30 pares históricos válidos com receita média positiva."
        )
    residual = r - (r.mean() / c.mean()) * c
    relative_variance = float(np.var(residual, ddof=1) / r.mean() ** 2)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return {
        "customers_per_arm": max(30, int(np.ceil(2 * z**2 * relative_variance / minimum_lift**2))),
        "minimum_relative_lift": minimum_lift,
        "power": power,
        "alpha": alpha,
        "relative_influence_variance": relative_variance,
    }
