# Contexto do Projeto — Nexar Dashcam Collision Prediction

## Objetivo do Projeto

Este projeto tem como objetivo desenvolver uma solução aplicada de inteligência artificial para prever risco de colisão ou quase-colisão a partir de vídeos de dashcam, utilizando o dataset da competição **Nexar Collision Prediction** disponível no Kaggle.

A ideia principal é criar uma pipeline completa envolvendo:

- análise exploratória dos vídeos;
- extração de frames;
- construção de um modelo baseline;
- evolução para modelos temporais;
- avaliação da capacidade de antecipar colisões;
- criação de um dashboard interativo para visualização dos resultados;
- documentação da metodologia para possível artigo científico.

O projeto deve ser tratado como uma solução de **engenharia aplicada com machine learning**, e não apenas como um notebook de competição Kaggle.

---

## Dataset

Fonte dos dados:

https://www.kaggle.com/competitions/nexar-collision-prediction/data

O dataset contém vídeos de dashcam com situações de direção normal, colisão ou quase-colisão. O objetivo é prever o risco de evento antes que ele ocorra, usando informação visual dos vídeos.

Os dados brutos não devem ser versionados no GitHub, pois são pesados e pertencem à competição/dataset original. Eles devem ficar localmente dentro da pasta:

```text
data/raw/
```

---

## Nome sugerido do repositório

Nome recomendado:

```text
nexar-dashcam-collision-prediction
```

Esse nome é claro, técnico e fácil de entender em contexto acadêmico, profissional ou de portfólio.

---

## Estratégia inicial do projeto

No início, a análise pode ficar concentrada em um único Jupyter Notebook. Isso facilita a exploração dos dados, testes rápidos e organização das ideias.

Notebook principal sugerido:

```text
notebooks/01_nexar_end_to_end_experiment.ipynb
```

Esse notebook pode conter:

1. Introdução do projeto;
2. Configuração dos caminhos;
3. Leitura dos metadados;
4. Análise exploratória dos dados;
5. Visualização de vídeos e frames;
6. Extração de frames;
7. Criação do dataset de treino;
8. Modelo baseline;
9. Treinamento;
10. Avaliação;
11. Análise de erros;
12. Exportação de resultados para o dashboard;
13. Conclusões.

A modularização em arquivos `.py` pode ser feita depois, quando as funções estiverem mais maduras.

---

## Estrutura inicial simplificada do repositório

```text
nexar-dashcam-collision-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_nexar_end_to_end_experiment.ipynb
│
├── outputs/
│   ├── figures/
│   └── predictions/
│
├── models/
│
└── app/
    └── streamlit_app.py
```

Essa estrutura é suficiente para começar. Depois, o projeto pode evoluir para uma estrutura mais profissional com `src/`, `scripts/`, `reports/` e módulos reutilizáveis.

---

## Estrutura futura mais profissional

Quando o projeto estiver mais maduro, pode evoluir para:

```text
nexar-dashcam-collision-prediction/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_frame_extraction.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   └── nexar_collision/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── evaluation/
│       └── visualization/
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_baseline.py
│   └── evaluate_model.py
│
├── app/
│   └── streamlit_app.py
│
├── outputs/
│   ├── figures/
│   ├── predictions/
│   └── submissions/
│
├── models/
│   └── checkpoints/
│
└── reports/
    ├── methodology.md
    └── paper_draft.md
```

---

## Pipeline técnico proposta

Fluxo geral do projeto:

```text
Vídeos do Kaggle
      ↓
Análise dos metadados
      ↓
Visualização dos vídeos e frames
      ↓
Extração de frames
      ↓
Construção de features visuais
      ↓
Modelo baseline
      ↓
Modelo temporal
      ↓
Avaliação da previsão antecipada
      ↓
Dashboard interativo
      ↓
Documentação para artigo
```

---

## Estratégia de modelagem

A recomendação é não começar diretamente com modelos complexos.

### Baseline inicial

Começar com um modelo simples e defensável:

```text
Frames extraídos dos vídeos
      ↓
CNN pré-treinada, como ResNet18 ou ResNet50
      ↓
Agregação das features dos frames
      ↓
Classificador binário
      ↓
Predição de risco de colisão
```

### Evolução posterior

Depois do baseline, testar modelos com componente temporal:

- CNN + LSTM;
- CNN + GRU;
- 3D CNN;
- Video Transformer;
- VideoMAE;
- TimeSformer.

Para artigo, é interessante mostrar a evolução:

```text
Modelo 1: frame único
Modelo 2: múltiplos frames com agregação simples
Modelo 3: CNN + LSTM/GRU
Modelo 4: modelo temporal avançado
```

---

## Papel do dashboard

O dashboard não deve ser apresentado como o centro científico do trabalho, mas sim como uma camada de visualização, análise e apoio à decisão.

Ele pode mostrar:

- vídeo analisado;
- probabilidade de colisão ao longo do tempo;
- frame crítico;
- classe real versus classe predita;
- falsos positivos;
- falsos negativos;
- comparação entre modelos;
- análise temporal do risco.

Tecnologia sugerida:

```text
Streamlit
```

Arquivo sugerido:

```text
app/streamlit_app.py
```

---

## Narrativa para artigo

O projeto pode ser apresentado como uma pesquisa aplicada em engenharia baseada em visão computacional e análise temporal de risco.

Texto-base para posicionamento do artigo:

> Este trabalho propõe uma pipeline aplicada para previsão antecipada de colisões a partir de vídeos de dashcam, combinando processamento de vídeo, modelos de visão computacional, avaliação temporal de risco e um dashboard interativo para suporte à análise dos resultados.

O foco do artigo não deve ser apenas o modelo, mas a combinação entre:

- processamento de vídeo;
- inteligência artificial aplicada;
- antecipação de risco;
- avaliação experimental;
- visualização dos resultados;
- potencial uso em segurança veicular.

---

## Ordem prática de execução

A ordem recomendada para iniciar o projeto é:

1. Criar o repositório;
2. Criar o README;
3. Criar o `.gitignore`;
4. Baixar os dados do Kaggle localmente;
5. Criar o notebook principal;
6. Fazer análise exploratória dos vídeos;
7. Extrair frames de exemplo;
8. Criar um baseline simples;
9. Avaliar os primeiros resultados;
10. Exportar predições para CSV;
11. Criar um dashboard simples;
12. Melhorar o modelo;
13. Documentar metodologia e resultados.

---

## Regra principal

No início:

```text
Um notebook bem organizado é suficiente.
```

Depois:

```text
Funções repetidas devem ser movidas para arquivos Python.
```

Por fim:

```text
O projeto deve evoluir para uma pipeline reprodutível, com dashboard e documentação científica.
```

---

## Requisitos iniciais sugeridos

Arquivo `requirements.txt` inicial:

```text
numpy
pandas
matplotlib
scikit-learn
opencv-python
tqdm
pillow
torch
torchvision
torchaudio
streamlit
plotly
python-dotenv
jupyter
kaggle
```

Dependências adicionais podem ser adicionadas posteriormente, como:

```text
timm
mlflow
transformers
pytorchvideo
datasets
```

---

## Resumo executivo

O projeto **Nexar Dashcam Collision Prediction** deve ser desenvolvido como uma solução de machine learning aplicada para previsão antecipada de colisões usando vídeos de dashcam. A fase inicial pode ser totalmente concentrada em um único Jupyter Notebook, com foco em exploração dos dados, extração de frames, baseline e avaliação. Posteriormente, o projeto deve evoluir para uma estrutura modular com scripts, dashboard em Streamlit e documentação voltada para publicação científica.

A principal contribuição esperada é uma pipeline prática que una visão computacional, modelagem temporal, avaliação de risco e visualização interativa para apoiar a análise de eventos críticos no trânsito.
