# Metodologia, resultados atuais e próximos passos

Este documento consolida, em português, a metodologia aplicada até o momento no
projeto **Nexar Dashcam Collision Prediction**, os principais resultados
obtidos e a sequência recomendada de trabalho.

## 1. Objetivo do projeto

O objetivo é construir um pipeline de predição de colisão a partir de vídeos de
dashcam. Em termos práticos, o modelo deve analisar frames ou sequências de
frames e emitir um alerta antes ou próximo do momento de risco, reduzindo:

- falsos alarmes em vídeos sem acidente;
- eventos perdidos em vídeos com acidente;
- alertas excessivamente tardios;
- decisões instáveis frame a frame.

Esse problema não é apenas uma classificação de vídeo. Ele também envolve uma
decisão temporal: **quando** alertar. Por isso, além de métricas comuns como
ROC-AUC, F1, precisão e recall, o projeto avalia métricas de alerta por vídeo,
como taxa de falso alarme, taxa de evento perdido e erro médio do tempo de
alerta.

## 2. Dados utilizados

A base local contém:

| Arquivo | Papel no projeto | Volume observado |
| --- | --- | ---: |
| `data/raw/train.csv` | Metadados de treino | 1.500 vídeos |
| `data/raw/test.csv` | Metadados de teste | 1.344 vídeos |
| `data/interim/sample_100_videos.csv` | Amostra estratificada para iteração rápida | 100 vídeos |
| `data/interim/sample_100_videos_splits.csv` | Split fixo treino/validação | 80 treino, 20 validação |
| `data/interim/temporal_frames_224_manifest.csv` | Frames temporais extraídos | 7.760 frames |

A amostra de 100 vídeos foi mantida balanceada: 50 vídeos sem evento e 50 vídeos
com evento. O split de validação possui 20 vídeos, sendo 10 positivos e 10
negativos.

O manifesto temporal contém 7.760 frames, dos quais 7.295 são negativos e 465
são positivos. Isso mostra um ponto importante do problema: mesmo com vídeos
balanceados, os frames positivos continuam sendo minoria, porque o trecho de
risco ocupa uma parte pequena do vídeo.

## 3. Metodologia aplicada

### 3.1 Estruturação do repositório

O projeto foi organizado com separação entre:

- `src/nexar_collision/`: código reutilizável de dados, modelos e avaliação;
- `scripts/`: comandos executáveis para treino, avaliação e sweeps;
- `notebooks/`: análise exploratória e revisão visual dos resultados;
- `reports/`: documentação técnica, resultados e próximos passos;
- `models/checkpoints/`: checkpoints locais dos modelos;
- `outputs/`: predições, curvas, tabelas e artefatos gerados;
- `mlflow.db`: banco local do MLflow, ignorado pelo Git.

Essa organização permite evoluir o projeto de forma reprodutível: cada
experimento importante fica conectado a código, relatório, notebook e registro
no MLflow.

### 3.2 Amostragem e split fixo

Foi criada uma amostra estratificada de 100 vídeos para acelerar a primeira fase
do projeto. Em seguida, foi criado um split fixo entre treino e validação:

- 80 vídeos para treino;
- 20 vídeos para validação;
- validação balanceada com 10 vídeos positivos e 10 negativos.

O split fixo é importante porque evita comparar experimentos em conjuntos
diferentes. A partir dele, os resultados passaram a ser avaliados no mesmo
recorte de validação.

### 3.3 Extração de frames

Os vídeos foram convertidos em frames redimensionados para 224x224 pixels. Essa
resolução é compatível com arquiteturas CNN clássicas como ResNet18.

No início, os experimentos usavam rótulo por vídeo: todos os frames de um vídeo
positivo herdavam o rótulo positivo. Isso serviu como baseline inicial, mas tem
uma limitação forte: frames muito anteriores ao acidente podem ser marcados como
positivos, confundindo o modelo.

### 3.4 Baseline por frame

O primeiro baseline treinou uma CNN para classificar frames individualmente. Ele
foi útil como prova de conceito, mas ainda não resolvia bem a lógica temporal de
alerta.

Resultado inicial observado em uma amostra de frames:

| Experimento | F1 frame-level | ROC-AUC frame-level | Leitura |
| --- | ---: | ---: | --- |
| Baseline inicial por frame | 0,609 | 0,560 | Sanity check, não suficiente para alerta real |

Esse resultado indicou que o pipeline funcionava, mas também mostrou que o
problema exigia rótulos temporais e avaliação por vídeo.

### 3.5 Rótulos temporais

A etapa seguinte foi trocar a lógica de rótulo por vídeo por uma lógica de
rótulo temporal. Em vez de marcar todos os frames de um vídeo positivo como
positivos, o pipeline marca como positivos apenas os frames próximos da região
de alerta/evento.

Essa mudança aproxima o treino do comportamento desejado: o modelo deve aprender
a subir o risco perto do evento, e não simplesmente reconhecer qualquer frame de
um vídeo que em algum momento terá colisão.

Essa etapa reduziu o ruído conceitual do treinamento e preparou o projeto para
avaliar alertas por vídeo.

### 3.6 Treino com best checkpoint e early stopping

Foi implementado suporte a:

- seleção automática do melhor checkpoint;
- monitoramento por métrica, principalmente ROC-AUC;
- `patience` para early stopping;
- registro de melhor época, melhores métricas e artefatos no MLflow.

Isso evita escolher apenas o último checkpoint treinado. Em problemas com poucos
dados e validação pequena, o último epoch pode não ser o melhor ponto de
generalização.

Experimentos principais:

| Modelo | Melhor época | Melhor ROC-AUC | Melhor F1 | ROC-AUC final | F1 final |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 do zero | 2 | 0,514 | 0,137 | 0,502 | 0,093 |
| ResNet18 pré-treinada | 3 | 0,599 | 0,058 | 0,558 | 0,078 |

A ResNet18 pré-treinada teve melhor ROC-AUC, mas o F1 ainda ficou baixo em nível
de frame. Isso reforçou que a decisão final deveria ser avaliada como alerta por
vídeo, não apenas como classificação isolada de frames.

### 3.7 Avaliação de alerta por vídeo

Para cada vídeo da validação, os scores por frame foram convertidos em decisão
de alerta. Foram avaliadas métricas como:

- **precisão**: entre os vídeos alertados, quantos eram realmente positivos;
- **recall**: entre os vídeos positivos, quantos receberam alerta;
- **false alarm rate**: proporção de vídeos negativos que receberam alerta;
- **missed event rate**: proporção de vídeos positivos sem alerta;
- **mean alert error**: diferença média entre o tempo do alerta e o tempo do
  evento.

Com threshold fixo em 0,50, os resultados ainda eram instáveis:

| Experimento | Precisão | Recall | Falso alarme | Evento perdido | Erro médio |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 do zero, best checkpoint | 0,533 | 0,800 | 0,700 | 0,200 | -14,366 s |
| ResNet18 pré-treinada, best checkpoint | 0,500 | 0,300 | 0,300 | 0,700 | -11,794 s |

O threshold 0,50 não era uma boa regra universal. Por isso, foi criado um sweep
de thresholds.

### 3.8 Sweep de thresholds

O sweep testa vários thresholds e seleciona pontos de operação conforme a meta
de recall. Um ponto de operação relevante foi exigir recall mínimo de 0,70.

Para a ResNet18 pré-treinada com melhor checkpoint, o melhor ponto bruto foi:

| Modelo | Regra | Threshold | Precisão | Recall | Falso alarme | Evento perdido | Erro médio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 pré-treinada | Score bruto | 0,23 | 0,667 | 0,800 | 0,400 | 0,200 | -7,616 s |

Esse resultado melhorou o equilíbrio entre recall e falso alarme, mas ainda
havia espaço para estabilizar a decisão temporal.

### 3.9 Agregação temporal simples

Antes de avançar para modelos sequenciais mais complexos, foi testada uma etapa
de pós-processamento temporal simples:

- média móvel dos scores;
- rolling max;
- alerta apenas após N frames consecutivos acima do threshold.

Essa etapa é importante porque decisões frame a frame tendem a oscilar. Em um
sistema de alerta, um único pico isolado pode gerar falso alarme. Exigir
consistência por alguns frames pode reduzir esse problema.

Resultados com recall mínimo de 0,70:

| Regra | Threshold | Precisão | Recall | Falso alarme | Evento perdido | Erro médio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Score bruto | 0,23 | 0,667 | 0,800 | 0,400 | 0,200 | -7,616 s |
| Média móvel 3s | 0,17 | 0,700 | 0,700 | 0,300 | 0,300 | -7,182 s |
| Média móvel 5s | 0,13 | 0,667 | 0,800 | 0,400 | 0,200 | -7,241 s |
| Rolling max 3s | 0,23 | 0,667 | 0,800 | 0,400 | 0,200 | -7,616 s |
| 2 frames consecutivos | 0,13 | 0,727 | 0,800 | 0,300 | 0,200 | -7,991 s |
| Média móvel 3s + 2 consecutivos | 0,12 | 0,667 | 0,800 | 0,400 | 0,200 | -9,116 s |

O melhor ponto atual é:

```text
Modelo: ResNet18 pré-treinada, best checkpoint
Pós-processamento: alerta após 2 frames consecutivos acima do threshold
Threshold: 0,13
Precisão: 0,727
Recall: 0,800
False alarm rate: 0,300
Missed event rate: 0,200
Mean alert error: -7,991 s
```

Esse é o melhor baseline atual porque reduziu a taxa de falso alarme de 0,400
para 0,300 mantendo recall de 0,800.

### 3.10 CNN + GRU/LSTM

Depois do baseline temporal simples, foi implementado um pipeline sequencial
com CNN + GRU/LSTM:

```text
sequência de frames -> ResNet18 -> sequência de embeddings -> GRU/LSTM -> score de alerta
```

Foram adicionados:

- dataset de sequências de frames;
- modelo `CnnRnnCollisionModel`;
- suporte a GRU e LSTM;
- treino com CNN pré-treinada congelada;
- avaliação por vídeo;
- sweep de thresholds;
- registro no MLflow.

Configuração inicial:

- sequência de 4 frames;
- ResNet18 pré-treinada congelada;
- GRU ou LSTM com hidden size 128;
- batch size 16;
- 4 épocas;
- monitoramento por ROC-AUC.

Resultados de treino/validação:

| Modelo sequencial | Melhor época | Melhor ROC-AUC | F1 final | ROC-AUC final | Early stopping |
| --- | ---: | ---: | ---: | ---: | --- |
| CNN + GRU seq4 | 4 | 0,567 | 0,142 | 0,567 | Não |
| CNN + LSTM seq4 | 1 | 0,549 | 0,100 | 0,540 | Sim, época 3 |

Avaliação em alerta com threshold 0,50:

| Modelo | Precisão | Recall | Falso alarme | Evento perdido | Erro médio |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN + GRU seq4 | 0,667 | 0,400 | 0,200 | 0,600 | -7,204 s |
| CNN + LSTM seq4 | 0,500 | 0,200 | 0,200 | 0,800 | -4,017 s |

Sweep com recall mínimo de 0,70:

| Modelo | Regra | Threshold | Precisão | Recall | Falso alarme | Evento perdido | Erro médio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline atual | 2 frames consecutivos | 0,13 | 0,727 | 0,800 | 0,300 | 0,200 | -7,991 s |
| CNN + GRU seq4 | Score bruto | 0,33 | 0,500 | 0,800 | 0,800 | 0,200 | -12,108 s |
| CNN + GRU seq4 | 2 frames consecutivos | 0,25 | 0,556 | 1,000 | 0,800 | 0,000 | -13,328 s |
| CNN + LSTM seq4 | Score bruto | 0,45 | 0,615 | 0,800 | 0,500 | 0,200 | -6,725 s |
| CNN + LSTM seq4 | 2 frames consecutivos | 0,43 | 0,583 | 0,700 | 0,500 | 0,300 | -9,047 s |

Conclusão desta etapa: o pipeline CNN + GRU/LSTM está implementado e funcional,
mas a primeira configuração testada ainda não superou o baseline com ResNet18
pré-treinada e regra de 2 frames consecutivos.

## 4. O que podemos concluir até agora

### 4.1 Conclusões positivas

O projeto já possui um pipeline completo e reproduzível:

- preparação de dados;
- extração de frames;
- rótulos temporais;
- treino de baseline CNN;
- seleção de melhor checkpoint;
- avaliação temporal por vídeo;
- sweeps de threshold;
- pós-processamento temporal;
- modelos sequenciais CNN + GRU/LSTM;
- tracking com MLflow;
- notebooks de revisão.

Também já existe um baseline razoável para a amostra atual:

```text
ResNet18 pré-treinada + best checkpoint + 2 frames consecutivos
```

Esse baseline é, até o momento, o ponto de comparação mais forte.

### 4.2 Conclusões técnicas

A troca de rótulo por vídeo para rótulo temporal foi uma melhoria metodológica
importante. Ela tornou o treinamento mais alinhado com o objetivo real do
projeto.

A seleção de best checkpoint também foi necessária, porque os resultados finais
do último epoch nem sempre são os melhores.

A agregação temporal simples foi muito eficiente em relação ao custo. A regra de
2 frames consecutivos melhorou a precisão e reduziu falso alarme sem reduzir o
recall.

A primeira versão CNN + GRU/LSTM validou a arquitetura e o pipeline, mas ainda
não trouxe ganho. Isso não significa que a abordagem sequencial seja ruim. A
configuração testada foi conservadora: sequência curta, encoder congelado e
poucos dados.

### 4.3 O que ainda não podemos afirmar

Ainda não é correto afirmar que o modelo está pronto para produção. A validação
tem apenas 20 vídeos, com 10 positivos e 10 negativos. Isso é útil para iterar,
mas pequeno para uma conclusão robusta.

Também não é correto afirmar que o threshold escolhido generaliza para todo o
dataset. O threshold foi selecionado na validação atual e precisa ser confirmado
em um holdout maior ou validação cruzada por vídeo.

Por fim, o modelo ainda apresenta falsos alarmes demais para um sistema de
segurança real. O melhor ponto atual tem false alarm rate de 0,300 na validação,
o que ainda é alto.

## 5. Como ver os resultados

### 5.1 MLflow

Para abrir a interface do MLflow:

```powershell
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Depois, acesse:

```text
http://127.0.0.1:5000
```

No MLflow é possível comparar runs, métricas, parâmetros, checkpoints e
artefatos dos experimentos.

### 5.2 Notebooks

Os notebooks principais para revisão são:

| Notebook | Uso |
| --- | --- |
| `notebooks/08_best_checkpoint_mlflow_review.ipynb` | Revisar best checkpoint e runs no MLflow |
| `notebooks/09_temporal_aggregation_review.ipynb` | Revisar agregação temporal e regras de alerta |

### 5.3 Relatórios técnicos

Os relatórios mais importantes são:

| Relatório | Conteúdo |
| --- | --- |
| `reports/best_checkpoint_experiment.md` | Resultados de best checkpoint e early stopping |
| `reports/temporal_aggregation_experiment.md` | Resultados de média móvel, rolling max e frames consecutivos |
| `reports/cnn_rnn_sequence_experiment.md` | Resultados da CNN + GRU/LSTM |
| `reports/next_steps_roadmap.md` | Roadmap técnico do projeto |

## 6. Próximos passos recomendados

### Prioridade 1: atacar o desbalanceamento temporal

O manifesto temporal tem muito mais frames negativos do que positivos. O próximo
experimento deve testar técnicas para lidar com esse desbalanceamento:

- `WeightedRandomSampler`;
- ponderação da loss;
- focal loss;
- amostragem mais forte de janelas próximas ao evento.

Esse passo pode melhorar tanto o baseline CNN quanto a CNN + GRU/LSTM.

### Prioridade 2: evoluir a CNN + GRU

Como a GRU foi melhor que a LSTM na primeira configuração, a próxima rodada deve
priorizar GRU:

- sequência de 8 frames;
- ResNet18 pré-treinada;
- descongelar parcialmente o último bloco da CNN;
- usar learning rates diferentes para CNN e GRU;
- manter avaliação com sweep e regra de frames consecutivos.

Uma configuração sugerida:

```text
CNN + GRU
sequence length: 8
CNN: ResNet18 pré-treinada
fine-tuning: último bloco liberado
loss: ponderada ou focal loss
avaliação: threshold sweep + 2 frames consecutivos
```

### Prioridade 3: ampliar a validação

Antes de tirar conclusões fortes, é importante ampliar a validação:

- usar mais vídeos;
- manter split por vídeo, nunca por frame;
- criar holdout fixo maior;
- avaliar estabilidade dos thresholds;
- comparar variação entre diferentes seeds.

Esse passo é essencial para saber se o ganho observado é real ou consequência
do tamanho pequeno da validação atual.

### Prioridade 4: melhorar a métrica de alerta

As métricas atuais já ajudam, mas o projeto pode evoluir para uma métrica mais
próxima do uso real:

- penalizar alerta muito tardio;
- penalizar alerta excessivamente cedo;
- tratar múltiplos alertas no mesmo vídeo;
- medir tempo até colisão no primeiro alerta válido;
- separar falso alarme leve de falso alarme crítico.

### Prioridade 5: criar uma revisão visual mais forte

O próximo notebook pode comparar visualmente:

- curva de risco por vídeo;
- threshold escolhido;
- instante do alerta;
- instante do evento;
- frames próximos ao alerta.

Isso ajudará a entender se o modelo está capturando sinais reais de risco ou se
está reagindo a artefatos visuais.

### Prioridade 6: preparar entrega/demo

Quando o modelo estiver mais estável, vale evoluir a camada de demonstração:

- dashboard Streamlit com upload/seleção de vídeo;
- curva de risco temporal;
- marcação do primeiro alerta;
- comparação entre modelos;
- resumo de métricas por experimento.

## 7. Próximo experimento sugerido

O próximo experimento mais útil é:

```text
Treinar CNN + GRU com sequência de 8 frames,
ResNet18 pré-treinada com fine-tuning parcial,
tratamento de desbalanceamento temporal,
e avaliar com sweep de thresholds + regra de 2 frames consecutivos.
```

Critério de sucesso:

- manter recall >= 0,80;
- reduzir false alarm rate abaixo de 0,30;
- melhorar precisão acima de 0,727;
- manter erro médio de alerta próximo ou melhor que -7,991 s.

Se esse experimento não superar o baseline, o baseline atual continua sendo a
melhor solução e a prioridade passa a ser escalar a validação e melhorar a
amostragem dos dados.

