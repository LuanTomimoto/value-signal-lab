# Metodologia e limites da interpretação

## 1. População e processo gerador

A base histórica possui uma linha por cliente adquirido. Canal, dispositivo, navegação anterior à compra, desconto e valor da primeira compra variam entre clientes. Uma propensão latente determina repetição de compras; receita futura e custos têm ruído multiplicativo. O gerador não usa dados de pessoas reais.

O experimento é uma amostra independente. A atribuição Bernoulli é 50/50 em expectativa, sem balanceamento forçado. O cenário padrão aumenta a receita futura em 12% e o custo em 2% no tratamento. Esses parâmetros são **hipóteses do gerador**, documentadas no JSON de proveniência, e não são aprendidos pelo modelo. A função `simulate_customers(..., experiment=True, effect=...)` permite cenários nulos e negativos para estudar sensibilidade.

O simulador não acopla o modelo preditivo a um mecanismo de leilão. Portanto, a análise preditiva e a demonstração de inferência experimental são componentes distintos. Nenhuma melhoria experimental é atribuível ao modelo treinado neste repositório.

## 2. Estimando de valor

O alvo individual é a **receita bruta acumulada nos primeiros 90 dias**, incluindo a primeira compra. A moeda da demonstração é BRL, escolhida como convenção de apresentação. Não modelamos margem, impostos, devoluções, custos operacionais ou valor além de 90 dias. O contrato exige receita acumulada não negativa e não decrescente em relação à primeira compra; bases com reembolsos precisam de outro contrato.

O modelo estima E[receita90 | informações da primeira compra]. Não estima efeito causal de aumentar um lance. `customer_id`, braço experimental, custos e receitas posteriores não são features. Features devem representar snapshots reais no momento da primeira compra; o software não consegue comprovar o horário em que um CSV externo foi coletado.

## 3. Maturidade e calendário

O corte padrão é 2026-11-01. Um cliente só tem alvo observado se aquisição + 90 dias ≤ corte. Rótulos futuros são mascarados mesmo se um CSV os preencher. Clientes já maduros com receita ausente causam erro em vez de exclusão silenciosa.

| Etapa | Aquisição de clientes | Uso |
| --- | --- | --- |
| Treino | 2024-01-01 a 2024-12-31 | Ajuste do modelo e baseline |
| Calibração | 2025-04-01 a 2025-06-30 | Ajuste global de escala |
| Teste | 2025-10-01 a 2026-07-31 | Avaliação final |

Janeiro–março e julho–setembro de 2025 funcionam como embargos entre etapas. A receita do último cliente de treino já amadureceu antes da primeira aquisição de calibração. O mesmo vale entre calibração e teste. A análise descritiva pode usar todo o histórico maduro; a modelagem usa somente os blocos declarados. O tamanho dos blocos e as datas efetivas são registrados em `summary.json`.

O gráfico de coortes exibe um horizonte somente quando **todos os clientes da coorte mensal** já o alcançaram. Assim, a composição da coorte não muda entre as células visíveis. Coortes sem janela completa permanecem vazias.

## 4. Modelo e avaliação

O pipeline aplica one-hot encoding às categorias e gradient boosting com perda Poisson à receita não negativa. A perda pode ser usada como função de regressão para alvos contínuos não negativos; não afirmamos que receita tem distribuição de contagem Poisson. O modelo usa número fixo de iterações e hiperparâmetros definidos antes da avaliação, sem seleção pelo teste ou validação aleatória interna.

Baseline: previsão = primeira compra × soma(receita90_treino) / soma(primeira_compra_treino).

Calibração: previsão_final = previsão_modelo × soma(receita90_calibração) / soma(previsão_calibração). Esse ajuste alinha o total da calibração; não garante calibração de todos os segmentos ou estabilidade no futuro.

Avaliação: MAE, RMSE, R² e viés agregado no teste temporal. Os decis comparam valor médio previsto e observado. O lift do decil superior é média observada do decil / média observada do teste; mede concentração e ordenação, não uplift causal. Empates são resolvidos pela ordem estável dos registros. Não usamos MAPE porque receitas pequenas e zeros tornam a métrica instável.

## 5. Experimento: ROAS e incerteza

Métrica primária por braço: ROAS = soma(receita90) / soma(custo). O efeito é o lift relativo ROAS_tratamento / ROAS_controle − 1. O intervalo usa bootstrap percentil bilateral de 95%, com 2.000 reamostragens independentes dos clientes de cada braço. Receita e custo são sempre reamostrados juntos. O controle precisa ter receita positiva em todas as reamostras; degeneração gera erro explícito.

SRM compara todos os clientes atribuídos aos dois braços contra proporção esperada 50/50, via qui-quadrado. Não restringimos esse diagnóstico aos clientes com janela madura. p < 0,01 bloqueia a interpretação automática. Um SRM sem sinal não prova randomização válida, ausência de viés ou independência.

Regra demonstrativa, definida antes da leitura: recomendar ganho prático apenas se limite inferior do IC > 5%; sinalizar perda se limite superior < 0; demais situações são inconclusivas para adoção. Trata-se de uma leitura única com horizonte fixo. Não há monitoramento sequencial, correção por consultas repetidas, decisão baseada em segmentos ou probabilidade bayesiana de vitória.

O teste de Welch sobre receita média é **secundário e exploratório**. Seu p-valor não testa ROAS e não controla a decisão primária. Intervalos bootstrap percentis podem ter cobertura inadequada em amostras pequenas ou caudas extremas; os limiares mínimos do código são validações operacionais, não garantias assintóticas.

## 6. Planejamento amostral

Usamos somente o histórico do bloco de treino como piloto, sem olhar o efeito experimental. Para R = receita e C = custo, com θ = média(R) / média(C), definimos resíduo W = R − θC. A variância relativa aproximada da razão é Var(W) / média(R)² por observação.

Com braços independentes de mesmo tamanho, a aproximação normal é:

```text
n por braço ≈ 2 × (z[1−α/2] + z[poder])² × Var(W) / (média(R)² × MDE_relativo²)
```

Padrões: α bilateral = 0,05; poder = 80%; MDE relativo = 5% **contra efeito zero**. Esse plano não garante 80% de probabilidade de obter limite inferior acima do limiar de adoção de 5%; isso exigiria planejar uma alternativa maior que esse limiar. Pressupomos a mesma distribuição de custos e variância entre braços para o planejamento. O custo diferente do tratamento no cenário é uma limitação dessa aproximação. Confirmar o plano por simulação para o desenho real, considerar clusterização e garantir 90 dias adicionais para maturação do último participante.

## 7. O que seria necessário em produção

O experimento supõe clientes independentes e custos atribuíveis a cada unidade. Plataformas reais têm interferência entre campanhas e aprendizado compartilhado. Condicionar análise somente a clientes convertidos pode criar seleção após o tratamento. Um teste de aquisição deve definir unidades elegíveis **antes da intervenção**, incluir não convertidos com receita zero e atribuir todo o gasto correspondente, ou usar campanhas/geografias e inferência na unidade de randomização apropriada.

A pipeline atual não implementa inferência por cluster, experimentos geográficos ou atribuição de gasto de não convertidos. Não se deve aplicar seu bootstrap individual a campanhas randomizadas como se os clientes fossem independentes. Para dados reais, também seriam necessários auditoria de atribuição, estabilidade temporal, análise de perdas de dados, definição de margem e desenho com orçamento e calendário registrados antes do lançamento.
