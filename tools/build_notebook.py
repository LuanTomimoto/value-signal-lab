"""Build the readable notebook; execute with --execute using the active Python kernel."""

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells = [
        md(
            "# Value Signal Lab\n## Valor do cliente e evidência experimental\n\n"
            "**Estudo independente de portfólio com dados inteiramente sintéticos.** "
            "Nenhum resultado mede desempenho comercial real.\n\n"
            "Pergunta: clientes mais baratos de adquirir também oferecem o maior retorno em 90 dias?"
        ),
        code(
            "import pandas as pd\nfrom IPython.display import display\n\n"
            "from value_signal_lab.data import (\n"
            "    TARGET,\n    channel_summary,\n    mature_customers,\n"
            "    simulate_customers,\n    validate_customers,\n)\n"
            "from value_signal_lab.experiment import analyze_experiment, plan_sample_size\n"
            "from value_signal_lab.model import fit_value_model, temporal_split\n\n"
            "SEED, AS_OF = 42, '2026-11-01'\n"
            "historical = validate_customers(simulate_customers(12000, SEED, AS_OF), AS_OF)\n"
            "mature = mature_customers(historical, AS_OF)\n"
            "display(pd.Series({'total': len(historical), 'maduros': len(mature), 'imaturos excluídos': len(historical)-len(mature)}))"
        ),
        md(
            "### 1. Distribuição e aquisição\nReceita acumulada não é lucro ou LTV vitalício. "
            "Custos e retorno por canal são descritivos; diferenças não identificam efeito causal do canal."
        ),
        code(
            "display(mature[[TARGET, 'first_order_revenue', 'acquisition_cost']].describe(percentiles=[.5, .9, .95, .99]).round(2))\n"
            "channels = channel_summary(mature)\ndisplay(channels.round(2))\n"
            "top20 = mature.nlargest(int(len(mature)*.2), TARGET)[TARGET].sum() / mature[TARGET].sum()\n"
            "print(f'Os 20% de maior receita concentram {top20:.1%} do valor observado.')"
        ),
        md(
            "### 2. Validação temporal e calibração\nTreino em 2024, calibração em abril–junho/2025, "
            "teste em outubro/2025–julho/2026. Embargos permitem observar os 90 dias antes da próxima etapa. "
            "Apenas informações disponíveis na primeira compra entram como features.\n\n"
            "O modelo é comparado com primeira compra × multiplicador estimado no treino. "
            "Ajuste multiplicativo global é aprendido na calibração, sem consultar o teste."
        ),
        code(
            "metrics, scored, deciles = fit_value_model(mature, SEED)\n"
            "display(pd.DataFrame({k: metrics[k] for k in ['baseline', 'model']}).round(3))\n"
            "display(pd.DataFrame({'clientes': metrics['split_sizes'], 'datas': metrics['split_dates']}))\n"
            "display(deciles.round(2))\n"
            "print(f'Lift do decil superior: {metrics[\"top_decile_lift\"]:.2f}x. Métrica de ordenação, não efeito incremental.')"
        ),
        md(
            "### 3. Experimento independente e incerteza\nO cenário assume +12% de receita **posterior à primeira compra** "
            "e +2% de custo no tratamento. Não usa o modelo treinado para selecionar clientes. "
            "Portanto, não demonstra que o modelo causou melhoria.\n\n"
            "Métrica primária: soma(receita) / soma(custo). Bootstrap reamostra clientes preservando pares de receita/custo. "
            "Decisão única pré-definida: limite inferior do IC 95% acima de +5%, após diagnóstico de alocação."
        ),
        code(
            "experiment = validate_customers(simulate_customers(6000, SEED+1, AS_OF, experiment=True), AS_OF, experiment=True)\n"
            "ab, lifts = analyze_experiment(experiment, n_boot=2000, seed=SEED)\n"
            "display(pd.Series({k: ab[k] for k in ['control_roas', 'treatment_roas', 'relative_lift', 'ci95', 'srm_pvalue', 'mature_counts']}))\n"
            "print(ab['decision'])"
        ),
        md(
            "### 4. Planejar antes de testar\nA aproximação para tamanho amostral usa variância da métrica de razão "
            "no histórico de treino, com poder de 80%, α bilateral de 5% e MDE relativo de 5% contra efeito zero. "
            "Isso não garante poder de 80% para que o limite inferior supere o limiar de adoção de 5%. "
            "O resultado é por braço e pressupõe clientes independentes, distribuições semelhantes e leitura com janela madura."
        ),
        code("plan = plan_sample_size(temporal_split(mature)['train'])\ndisplay(pd.Series(plan))"),
        md(
            "### 5. Interpretação e próximos estudos\n"
            "- Boa previsão não prova que um sinal melhora leilões. Isso exige intervenção randomizada real.\n"
            "- Comparações entre canais podem refletir seleção de clientes e diferenças de mix.\n"
            "- Clientes de campanhas reais podem ser dependentes; inferência deve respeitar a unidade de randomização.\n"
            "- Não convertidos e todo o gasto elegível devem entrar no estimando de um teste de aquisição real.\n"
            "- Próximos estudos: estabilidade fora do período, margem, calibração por segmento e experimentos por cluster.\n\n"
            "Gráficos interativos completos: abra `reports/demo/report.html` localmente. "
            "Veja também [metodologia](../docs/methodology.md) e [resultados](../reports/demo/README.md)."
        ),
    ]
    for i, cell in enumerate(cells):
        cell["id"] = f"value-signal-{i:02d}"
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": sys.version.split()[0]}
    if "--execute" in sys.argv:
        NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
    target = ROOT / "notebooks" / "01_value_signal_analysis.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, target)
    print(target)


if __name__ == "__main__":
    main()
