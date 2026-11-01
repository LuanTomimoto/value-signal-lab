# Contrato de dados

Use CSV UTF-8, vírgula como separador, ponto decimal e datas ISO (`YYYY-MM-DD`). Cada linha deve representar **um cliente independente**. O histórico e o experimento são arquivos separados.

| Coluna | Tipo | Regra |
| --- | --- | --- |
| `customer_id` | string | Identificador pseudonimizado, único, obrigatório |
| `acquired_at` | data | Primeira compra, sem timezone; não posterior ao corte |
| `channel` | string | Categoria de aquisição; demo usa Google, Meta, TikTok |
| `device` | string | Dispositivo na primeira compra |
| `sessions_before_purchase` | número | Sessões acumuladas até a primeira compra, não negativo |
| `product_views_before_purchase` | número | Visualizações até a primeira compra, não negativo |
| `discount_rate` | número | Desconto da primeira compra, entre 0 e 1 |
| `first_order_revenue` | número | Receita da primeira compra, não negativa |
| `acquisition_cost` | número | Custo por cliente, estritamente positivo |
| `revenue_90d` | número | Receita acumulada, incluindo primeira compra; obrigatório se maduro |
| `arm` | string | Obrigatório no experimento: `control` ou `value_signal` |
| `revenue_30d`, `revenue_60d` | número | Opcionais; acumulados, usados apenas na demo de coortes |

Nenhuma coluna deve conter nome, e-mail, telefone ou outro identificador pessoal direto. O projeto não precisa dessas informações. A unicidade é verificada dentro de cada arquivo; a independência das amostras e a atribuição randomizada precisam ser verificadas na origem.

O contrato é voltado a receita acumulada sem devoluções. Valores finitos não negativos são obrigatórios nas colunas numéricas; features e custos ausentes geram erro. Alvos maduros menores que a primeira compra também geram erro. Receita de 90 dias para clientes imaturos é ignorada e mascarada, mesmo se estiver preenchida.

Para usar o modelo com CSV próprio, o histórico deve cobrir os blocos fixos de treino (2024), calibração (abril–junho/2025) e teste (outubro/2025–julho/2026), com pelo menos 100 clientes maduros por bloco. Para outro calendário, altere explicitamente `temporal_split` e preserve o embargo mínimo de 90 dias. O experimento precisa de ao menos 30 clientes maduros em cada braço; o tamanho mínimo não garante poder estatístico.

```powershell
New-Item -ItemType Directory -Force data/raw
# Coloque os dois CSVs nessa pasta; ela é ignorada pelo Git.
.\.venv\Scripts\value-signal-lab.exe --historical-csv data/raw/customers.csv --experiment-csv data/raw/experiment.csv --output reports/local
```

Com CSV próprio, as saídas são `summary.json`, `channels.csv` e `calibration.csv`. O HTML e a narrativa de simulação são gerados somente para a demonstração sintética, para não rotular dados externos de forma incorreta. Use uma pasta de saída nova ao trocar de fonte; não reutilize um relatório de uma execução anterior como resultado atual.

O programa não envia dados a terceiros. `reports/local/` e `data/raw/` são ignorados pelo Git; outras pastas não recebem essa proteção. A pasta pública `reports/demo/` é reservada aos artefatos sintéticos e o CLI impede sobrescrevê-la diretamente com CSV próprio.
