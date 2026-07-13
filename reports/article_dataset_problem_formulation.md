# 2. Conjunto de Dados e Formulação do Problema

## 2.1 Conjunto de Dados Nexar

O conjunto de dados utilizado neste trabalho é o **Nexar Dashcam Collision Prediction**, composto por vídeos obtidos a partir de câmeras veiculares instaladas em automóveis. O conjunto de treino disponibilizado contém 1.500 vídeos anotados, divididos de forma balanceada entre exemplos negativos, nos quais não há colisão ou quase-colisão, e exemplos positivos, nos quais ocorre um evento de risco. No conjunto analisado, há 750 vídeos com `target = 0` e 750 vídeos com `target = 1`.

Cada vídeo é identificado por um campo `id` e possui, no arquivo de anotações, a classe associada ao vídeo. Para os exemplos positivos, também são fornecidos dois marcos temporais relevantes: `time_of_event`, que indica o instante do evento de colisão ou quase-colisão, e `time_of_alert`, que representa o primeiro instante em que um observador atento poderia reconhecer que a situação estava se tornando perigosa. Para os exemplos negativos, esses campos temporais não são preenchidos, pois não há evento de risco anotado.

Além do conjunto de treino, o desafio também disponibiliza 1.344 vídeos de teste sem rótulos públicos. Neste trabalho, os experimentos de desenvolvimento e validação são conduzidos sobre o conjunto de treino anotado, enquanto o conjunto de teste é preservado para submissões ou avaliações externas associadas ao desafio.

## 2.2 Definição da Tarefa

A tarefa investigada consiste em antecipar a ocorrência de colisões ou quase-colisões a partir de vídeos de *dashcam*. Diferentemente de uma classificação convencional de vídeo, na qual o objetivo é apenas determinar se um evento ocorre ou não em algum momento da sequência, a antecipação de acidentes exige estimar a evolução temporal do risco antes da ocorrência efetiva do evento.

Assim, a saída desejada do modelo não é apenas uma classe final para o vídeo, mas uma curva de risco ao longo do tempo. Para cada instante analisado, o modelo estima uma probabilidade de risco. A partir dessa curva, um sistema de alerta pode ser definido por meio de um limiar de decisão e de uma regra temporal de pós-processamento, como exigir que o risco permaneça acima do limiar por um número mínimo de frames consecutivos.

Formalmente, dado um vídeo \(V_i\) composto por uma sequência de frames ou amostras temporais, o objetivo é estimar uma função de risco:

```text
r_i(t) = P(y_i(t) = 1 | V_i, t)
```

em que \(r_i(t)\) representa a probabilidade estimada de que o vídeo esteja em uma região temporal associada a risco de colisão no instante \(t\). O alerta é emitido quando a curva \(r_i(t)\) satisfaz uma regra de decisão definida no conjunto de validação.

Essa formulação permite avaliar não apenas se o modelo identifica corretamente vídeos positivos, mas também se ele emite alertas com antecedência suficiente e com uma taxa aceitável de falsos alarmes.

## 2.3 Formulação Temporal dos Rótulos

Para transformar a tarefa de vídeo em uma tarefa temporal supervisionada, os vídeos são amostrados em uma taxa fixa de frames por segundo. Nos experimentos conduzidos até o momento, foram utilizadas amostragens de 1 FPS e 2 FPS, com frames redimensionados para 224 x 224 pixels.

Nos vídeos negativos, todos os frames recebem rótulo temporal negativo, pois não há colisão ou quase-colisão anotada. Nos vídeos positivos, os rótulos temporais são definidos a partir dos campos `time_of_alert` e `time_of_event`. A região de risco é considerada como o intervalo entre um instante anterior ao alerta anotado e o momento do evento:

```text
alert_start = max(0, time_of_alert - pre_alert_margin)
```

```text
y_i(t) = 1 se alert_start <= t <= time_of_event
y_i(t) = 0 caso contrário
```

Essa estratégia transforma a anotação original em um problema de classificação temporal por frame ou por janela causal de frames. O parâmetro `pre_alert_margin` permite incluir uma margem anterior ao instante de alerta humano, incentivando o modelo a aprender sinais visuais que antecedem a percepção explícita do risco.

A formulação temporal também torna possível construir curvas de risco e avaliar diferentes regras de alerta. Em vez de depender apenas da probabilidade média ou máxima do vídeo, o modelo pode ser analisado em relação ao instante em que o primeiro alerta é emitido, à antecedência desse alerta e aos falsos positivos produzidos em vídeos negativos.

## 2.4 Protocolo Experimental Inicial

Os primeiros experimentos foram conduzidos sobre uma amostra estratificada de 100 vídeos, composta por 50 vídeos positivos e 50 vídeos negativos. Essa amostra permitiu validar rapidamente a pipeline de processamento, extração de frames, treinamento, avaliação temporal e rastreamento de experimentos com MLflow.

Sobre essa amostra inicial, foi criado um split por vídeo com 80 vídeos para treino e 20 vídeos para validação. A separação por vídeo é essencial para evitar vazamento de dados, pois frames ou janelas derivados do mesmo vídeo não devem aparecer simultaneamente nos conjuntos de treino e validação.

Essa etapa inicial permitiu comparar diferentes abordagens:

- modelo convolucional baseado em ResNet18 treinado em frames temporais;
- uso de pesos pré-treinados em ImageNet;
- seleção do melhor checkpoint por métrica de validação;
- pós-processamento temporal por média móvel, máximo móvel e frames consecutivos;
- modelos sequenciais CNN + GRU/LSTM;
- estratégias de tratamento de desbalanceamento, como `WeightedRandomSampler` e focal loss.

Os melhores resultados nessa amostra foram obtidos pelo modelo ResNet18 pré-treinado com seleção de melhor checkpoint e regra de alerta baseada em dois frames consecutivos acima do limiar. Entretanto, como essa validação utilizava apenas 20 vídeos, seus resultados são tratados como evidência experimental inicial, e não como prova suficiente de confiabilidade para uso em produto.

## 2.5 Protocolo de Consolidação em Escala de Produto

Para aproximar a avaliação das exigências de um sistema aplicável em produto, foi criado um split completo a partir dos 1.500 vídeos anotados do conjunto de treino. Esse split separa os dados em treino, validação e holdout, mantendo o balanceamento entre vídeos positivos e negativos.

A divisão adotada é:

| Split | Vídeos negativos | Vídeos positivos | Total |
| --- | ---: | ---: | ---: |
| Treino | 526 | 526 | 1052 |
| Validação | 112 | 112 | 224 |
| Holdout | 112 | 112 | 224 |

O conjunto de treino é utilizado para ajustar os parâmetros dos modelos. O conjunto de validação é utilizado para seleção de checkpoint, escolha de limiar, calibração e comparação entre configurações. O conjunto de holdout deve ser usado apenas após a escolha de um candidato final, funcionando como uma avaliação independente da capacidade de generalização.

Essa separação evita que o limiar de alerta e as decisões de arquitetura sejam ajustados diretamente sobre o conjunto usado para a conclusão final. Dessa forma, o desempenho em holdout fornece uma estimativa mais confiável do comportamento esperado em vídeos não vistos.

## 2.6 Critérios de Avaliação

A avaliação dos modelos considera dois níveis complementares: métricas de classificação temporal e métricas de alerta em vídeo.

No nível dos frames ou janelas temporais, são utilizadas métricas como:

- acurácia;
- precisão;
- revocação;
- F1-score;
- área sob a curva ROC.

Essas métricas permitem medir se o modelo consegue distinguir regiões temporais de risco de regiões normais. No entanto, elas não são suficientes para avaliar a utilidade prática do sistema, pois um modelo pode apresentar boa ordenação de scores e ainda assim produzir alertas inadequados quando aplicado a vídeos completos.

Por esse motivo, a avaliação principal deste trabalho considera métricas de alerta em vídeo:

- `alert_precision`: proporção de alertas emitidos que correspondem a vídeos positivos;
- `alert_recall`: proporção de vídeos positivos em que o sistema emitiu alerta;
- `false_alarm_rate`: proporção de vídeos negativos em que o sistema emitiu alerta;
- `missed_event_rate`: proporção de vídeos positivos sem alerta;
- `mean_alert_time_error`: diferença média entre o alerta previsto e o alerta anotado;
- antecedência média ou mediana do alerta em relação ao evento.

Essas métricas refletem melhor o compromisso central da tarefa: emitir alertas suficientemente cedo sem gerar uma quantidade excessiva de falsos positivos. Em aplicações de segurança veicular, falsos negativos representam eventos perigosos não detectados, enquanto falsos positivos frequentes podem reduzir a confiança do usuário no sistema.

## 2.7 Critérios para Modelo Candidato a Produto

Como o objetivo do projeto inclui a construção de uma solução aplicável em produto, os critérios de aceitação do modelo são mais rigorosos do que os critérios usados apenas para comparação experimental.

No conjunto de validação, um modelo candidato deve buscar pelo menos:

```text
alert_recall >= 0.80
false_alarm_rate <= 0.25
alert_precision >= 0.75
```

No conjunto de holdout, utilizado apenas para avaliação final do candidato selecionado, o critério mínimo adotado é:

```text
alert_recall >= 0.80
false_alarm_rate <= 0.30
alert_precision >= 0.70
```

Esses valores não representam garantias absolutas de segurança, mas estabelecem uma régua inicial para decidir se um modelo é suficientemente promissor para ser integrado a uma interface de produto. Caso o modelo não atinja esses critérios no holdout, ele deve ser tratado como resultado experimental e não como modelo final.

## 2.8 Relação com a Interface de Produto

A formulação temporal da tarefa também orienta a construção da interface web do produto. Como o modelo gera uma curva de risco ao longo do vídeo, a interface deve permitir que o usuário visualize não apenas a classe predita, mas também:

- a evolução temporal da probabilidade de risco;
- o limiar de alerta selecionado;
- o instante em que o alerta foi emitido;
- o instante anotado de alerta e de evento, quando disponíveis;
- os trechos de maior risco;
- os casos de falso positivo e falso negativo.

Dessa forma, a interface não é apenas uma camada visual, mas uma ferramenta de inspeção e auditoria do comportamento temporal do modelo. Essa característica é especialmente importante em um domínio de segurança, no qual a interpretabilidade operacional e a análise dos erros são componentes essenciais para a evolução do sistema.
