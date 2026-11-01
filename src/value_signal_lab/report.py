"""Self-contained, offline HTML artifact with computed findings and interactive plots."""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from .data import TARGET

COLORS = ["#7367f0", "#28aa95", "#f2ae57"]


def chart(fig):
    fig.update_layout(
        template="plotly_white",
        font={"family": "Arial", "color": "#55556f", "size": 12},
        colorway=COLORS,
        margin={"l": 45, "r": 25, "t": 25, "b": 45},
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.13},
    )
    fig.update_xaxes(gridcolor="#eeeef5")
    fig.update_yaxes(gridcolor="#eeeef5")
    return fig.to_html(
        full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True}
    )


def render_report(
    historical, mature, channels, metrics, scored, deciles, experiment, lifts, plan, output, as_of
):
    revenue = mature[TARGET].sum()
    spend = mature["acquisition_cost"].sum()
    values = np.sort(mature[TARGET].to_numpy())
    top_share = values[-int(np.ceil(len(values) * 0.2)) :].sum() / values.sum()
    model_gain = 1 - metrics["model"]["mae"] / metrics["baseline"]["mae"]

    def pct(x):
        return f"{100 * x:.1f}%"

    def money(x):
        return f"R$ {x:,.0f}"

    channel_fig = go.Figure()
    for metric, label, color in [
        ("cac", "CAC", COLORS[0]),
        ("mean_value_90d", "Valor médio 90d", COLORS[1]),
    ]:
        channel_fig.add_bar(
            x=channels["channel"], y=channels[metric], name=label, marker_color=color
        )
    channel_fig.update_yaxes(title="R$ por cliente")

    concentration = go.Figure()
    concentration.add_scatter(
        x=np.linspace(0, 100, len(values) + 1),
        y=np.r_[0, np.cumsum(values) / values.sum() * 100],
        fill="tozeroy",
        line_color=COLORS[0],
        name="Receita acumulada",
    )
    concentration.add_scatter(
        x=[0, 100], y=[0, 100], line={"dash": "dot", "color": "#acacbb"}, name="Igualdade"
    )
    concentration.update_xaxes(title="Clientes ordenados por valor (%)")
    concentration.update_yaxes(title="Receita acumulada (%)")

    calibration = go.Figure()
    for col, label, color in [
        ("predicted", "Previsto", COLORS[0]),
        ("observed", "Observado", COLORS[1]),
    ]:
        calibration.add_scatter(
            x=deciles["decile"], y=deciles[col], name=label, mode="lines+markers", line_color=color
        )
    calibration.update_xaxes(title="Decil de valor previsto · teste temporal", dtick=1)
    calibration.update_yaxes(title="Receita média em 90 dias (R$)")

    cohorts = historical.copy()
    cohorts["cohort"] = cohorts["acquired_at"].dt.strftime("%Y-%m")
    latest = cohorts.groupby("cohort")["acquired_at"].max()
    cohort_data = []
    for label, column, horizon in [
        ("Dia 0", "first_order_revenue", 0),
        ("30 dias", "revenue_30d", 30),
        ("60 dias", "revenue_60d", 60),
        ("90 dias", TARGET, 90),
    ]:
        if column in cohorts:
            observed = cohorts.groupby("cohort")[column].mean()
            # Only show an age if the ENTIRE monthly cohort has reached it.
            observed.loc[latest + pd.Timedelta(days=horizon) > pd.Timestamp(as_of)] = np.nan
            cohort_data.append(observed.rename(label))
    cohort_matrix = pd.concat(cohort_data, axis=1).tail(12)
    cohort_fig = px.imshow(
        cohort_matrix,
        aspect="auto",
        color_continuous_scale=["#f1effe", "#7367f0"],
        labels={"x": "Horizonte", "y": "Coorte de aquisição", "color": "R$ / cliente"},
    )
    cohort_fig.update_xaxes(side="bottom")

    bootstrap = go.Figure(
        go.Histogram(x=lifts * 100, nbinsx=65, marker_color=COLORS[0], name="Reamostras")
    )
    bootstrap.add_vline(x=0, line_color="#a7a7b8", line_dash="dot")
    bootstrap.add_vline(
        x=experiment["minimum_practical_lift"] * 100,
        line_color=COLORS[1],
        line_dash="dash",
        annotation_text="Limiar prático",
    )
    for bound in experiment["ci95"]:
        bootstrap.add_vline(x=100 * bound, line_color=COLORS[2], line_dash="dot")
    bootstrap.update_xaxes(title="Lift relativo de ROAS (%)")
    bootstrap.update_yaxes(title="Frequência bootstrap")

    errors = go.Figure()
    for name, label, color in [
        ("baseline", "Primeira compra × multiplicador", "#bcb6f7"),
        ("model", "Modelo pLTV calibrado", COLORS[0]),
    ]:
        errors.add_bar(
            x=["MAE", "RMSE"],
            y=[metrics[name]["mae"], metrics[name]["rmse"]],
            name=label,
            marker_color=color,
        )
    errors.update_yaxes(title="Erro em R$ · menor é melhor")

    def panel(number, title, description, figure, extra="", wide=False):
        return (
            f'<section class="panel {"wide" if wide else ""}"><div class="eyebrow">ANÁLISE {number}</div>'
            f"<h2>{title}</h2><p>{description}</p>{chart(figure)}{extra}</section>"
        )

    rows = "".join(
        f'<tr><td><span class="dot" style="background:{COLORS[i]}"></span>{escape(str(row.channel))}</td>'
        f"<td>{row.customers:,}</td><td>{money(row.cac)}</td><td>{money(row.mean_value_90d)}</td>"
        f"<td><strong>{row.roas_90d:.2f}×</strong></td></tr>"
        for i, row in enumerate(channels.itertuples())
    )
    table = (
        '<div class="table-wrap"><table><thead><tr><th>Canal</th><th>Clientes</th><th>CAC</th><th>Valor 90d</th><th>ROAS</th></tr></thead><tbody>'
        + rows
        + "</tbody></table></div>"
    )
    ci = experiment["ci95"]
    cards = [
        (
            "Receita observada · 90 dias",
            money(revenue),
            f"{len(mature):,} clientes com janela completa",
        ),
        (
            "ROAS da base histórica",
            f"{revenue / spend:.2f}×",
            f"{money(spend)} em aquisição simulada",
        ),
        (
            "Lift de ROAS · experimento",
            pct(experiment["relative_lift"]),
            f"IC 95%: {pct(ci[0])} a {pct(ci[1])}",
        ),
        (
            "Redução de MAE vs. baseline",
            pct(model_gain),
            "Avaliação fora da amostra · teste temporal",
        ),
    ]
    cards_html = "".join(
        f'<div class="kpi"><div class="kpi-label">{a}</div><div class="kpi-value">{b}</div><div class="kpi-detail">{c}</div></div>'
        for a, b, c in cards
    )
    content = "".join(
        [
            panel(
                "01",
                "Aquisição barata ou aquisição valiosa?",
                "Compare o custo de aquisição com a receita acumulada por cliente. Diferenças entre canais são descritivas e não estimam causalidade.",
                channel_fig,
                table,
            ),
            panel(
                "02",
                "O valor se concentra em poucos clientes",
                f"Os 20% de maior valor representam {pct(top_share)} da receita histórica. A heterogeneidade motiva investigar sinais de valor.",
                concentration,
            ),
            panel(
                "03",
                "O sinal ordena e calibra o valor?",
                f"Decis superiores devem concentrar receita observada. O decil superior tem {metrics['top_decile_lift']:.2f}× o valor médio no teste; isso mede ordenação, não retorno incremental.",
                calibration,
            ),
            panel(
                "04",
                "Melhor que uma regra simples?",
                f"Comparação na mesma amostra temporal. R² do modelo: {metrics['model']['r2']:.3f}; viés agregado: {metrics['model']['aggregate_bias_pct']:+.1f}%. O ajuste global não garante calibração por segmento.",
                errors,
            ),
            panel(
                "05",
                "Valor que chega depois da conversão",
                "Receita média acumulada nas 12 coortes mais recentes. Células sem janela completa para toda a coorte ficam vazias. Nenhum valor futuro é tratado como observado.",
                cohort_fig,
            ),
            panel(
                "06",
                "Quanto do ganho resiste à incerteza?",
                f"{experiment['bootstrap_replicates']:,} reamostragens por cliente preservam o par receita/custo. ROAS controle: {experiment['control_roas']:.2f}×; tratamento: {experiment['treatment_roas']:.2f}×. Intervalo percentil, leitura única.",
                bootstrap,
            ),
        ]
    )
    split = metrics["split_sizes"]
    html = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Value Signal Lab · Análise de valor do cliente</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#f7f8fc;color:#24243c;font-family:Arial,sans-serif;font-size:14px}}
header{{background:#fff;border-bottom:1px solid #e9e9f2;padding:22px 5%;display:flex;justify-content:space-between;align-items:center}}
.brand{{font-size:18px;font-weight:750;letter-spacing:-.5px}} .brand b{{background:#7367f0;color:white;padding:8px 10px;border-radius:10px;margin-right:12px}}
.pill{{background:#eeebff;color:#6053ce;border-radius:30px;padding:8px 14px;font-size:11px;letter-spacing:.7px;font-weight:bold}}
main{{max-width:1440px;margin:auto;padding:42px 4%}} .eyebrow{{font-size:10px;font-weight:bold;letter-spacing:1.7px;color:#8c84b0;margin-bottom:12px}}
h1{{font-size:38px;letter-spacing:-1.5px;margin:8px 0 14px;line-height:1.12}} h1 span{{color:#7367f0}}
.intro{{max-width:760px;color:#717187;line-height:1.7;font-size:15px}} .meta{{color:#858599;font-size:12px;margin:22px 0 28px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin:28px 0}}
.kpi,.panel,.readout{{background:#fff;border:1px solid #e9e9f2;border-radius:14px;padding:24px}}
.kpi-label{{font-size:12px;color:#77778f}} .kpi-value{{font-size:30px;font-weight:750;letter-spacing:-1px;margin:13px 0 9px}}
.kpi-detail{{font-size:11px;color:#9090a4;line-height:1.5}} .readout{{border-left:4px solid #7367f0;margin-bottom:26px;line-height:1.8}}
.readout strong{{display:block;color:#6053ce}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}
.panel{{min-width:0;overflow:hidden}} h2{{font-size:19px;letter-spacing:-.4px;margin:0 0 12px}} .panel p{{color:#838399;font-size:12px;line-height:1.7;min-height:60px;margin:0 0 12px}}
.table-wrap{{overflow-x:auto}} table{{width:100%;border-collapse:collapse;font-size:11px}} th{{color:#9999ac;font-weight:normal;text-align:left;padding:12px 6px}}
td{{border-top:1px solid #f0f0f6;padding:12px 6px}} .dot{{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:8px}}
.methods{{background:#efedf9;border-radius:14px;padding:28px;margin-top:26px;line-height:1.9;color:#696581;font-size:12px}}
.methods h2{{color:#4b466c}} .methods ul{{padding-left:20px}} a{{color:#7367f0}} footer{{padding:24px 0;color:#9999ac;font-size:11px;line-height:1.8}}
@media(max-width:1000px){{.kpis{{grid-template-columns:1fr 1fr}} .grid{{grid-template-columns:1fr}} h1{{font-size:32px}}}}
@media(max-width:550px){{header{{padding:18px}} .brand{{font-size:14px}} .pill{{font-size:9px}} main{{padding:26px 14px}} .kpis{{gap:10px}} .kpi{{padding:16px}} .kpi-value{{font-size:23px}} .panel{{padding:18px}}}}
</style><script>{get_plotlyjs()}</script></head><body>
<header><div class="brand"><b>v.</b>Value Signal Lab</div><div class="pill">ESTUDO INDEPENDENTE · DADOS SINTÉTICOS</div></header>
<main><div class="eyebrow">ESTATÍSTICA APLICADA À AQUISIÇÃO DE CLIENTES</div>
<h1>Mais valor por cliente.<br><span>Mais evidência por decisão.</span></h1>
<p class="intro">Um laboratório reproduzível para investigar valor em 90 dias, qualidade de aquisição e incerteza experimental com dados sintéticos.</p>
<div class="meta">CORTE: {escape(as_of)} &nbsp; / &nbsp; MOEDA: BRL (SIMULADA) &nbsp; / &nbsp; HORIZONTE: 90 DIAS</div>
<div class="kpis">{cards_html}</div>
<div class="readout"><strong>Leitura do experimento</strong>{escape(experiment["decision"])}<br>
Limiar prático pré-definido: {pct(experiment["minimum_practical_lift"])}. SRM na alocação completa: p = {experiment["srm_pvalue"]:.3f}.
Este cenário foi gerado com uma hipótese de efeito; seus resultados não medem o desempenho real de nenhuma empresa.</div>
<div class="grid">{content}</div>
<div class="methods"><h2>Uma análise que explicita suas hipóteses</h2><ul>
<li><b>Alvo:</b> receita acumulada em 90 dias, incluindo a primeira compra. Não é lucro nem LTV vitalício.</li>
<li><b>Modelo:</b> gradient boosting com perda Poisson, apenas features da primeira conversão. Treino: {split["train"]:,}; calibração: {split["calibration"]:,}; teste: {split["test"]:,}. Embargos de 90 dias entre blocos; fator de calibração estimado antes do teste.</li>
<li><b>Experimento:</b> braços independentes com alocação 50/50 em expectativa; unidade sintética = cliente. ROAS = soma das receitas / soma dos custos. Bootstrap pareado dentro de cada braço; não é probabilidade bayesiana de vitória.</li>
<li><b>Planejamento:</b> aproximadamente {plan["customers_per_arm"]:,} clientes por braço para MDE de {pct(plan["minimum_relative_lift"])}, poder de {pct(plan["power"])} e α = {plan["alpha"]}. Aproximação baseada exclusivamente no histórico de treino.</li>
<li><b>Limite causal:</b> o simulador não reproduz leilões, seleção de convertidos, interferência ou aprendizado das plataformas. Em produção, a unidade de randomização e a inclusão de não convertidos precisam corresponder ao desenho do teste.</li>
</ul></div>
<footer>Projeto de portfólio em estatística aplicada.
Dados simulados, sem acesso a dados privados e sem envio a plataformas de anúncios. Gráficos disponíveis offline neste arquivo.</footer></main></body></html>"""
    Path(output).write_text(html, encoding="utf-8")
