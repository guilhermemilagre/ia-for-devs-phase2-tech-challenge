# Classificador de Cancer de Mama - Tech Challenge FIAP IA For Devs

Pipeline completo de classificacao supervisionada para o dataset **Breast Cancer Wisconsin**,
desenvolvido como Tech Challenge das Fases 1 e 2 do curso IA For Devs (FIAP).

Classifica tumores de mama como **Maligno** ou **Benigno** a partir de 30 features morfologicas
extraidas por aspiracao por agulha fina (PAAF).

**Dataset:** [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) -
569 amostras, 30 features numericas, 2 classes (357 benignos / 212 malignos).
Salvar o arquivo baixado em `data/data.csv`.

---

## Estrutura do projeto

```
.
├── notebooks/
│   └── breast-cancer-classification.ipynb  # Pipeline completo (145 celulas)
├── api/
│   └── main.py                             # FastAPI - /predict e /interpret
├── app/
│   └── streamlit_app.py                    # Interface web Streamlit
├── models/
│   ├── breast_cancer_svm.joblib
│   ├── breast_cancer_regressao_logistica.joblib
│   ├── breast_cancer_svm_ag.joblib
│   └── breast_cancer_regressao_logistica_ag.joblib
├── logs/
│   ├── ga_experiments.log                  # Log estruturado dos experimentos
│   ├── figures/                            # Graficos gerados pelo notebook
│   └── llm_outputs/                        # Respostas do LLM salvas em .md
├── data/
│   └── data.csv                            # Dataset (baixar do link acima)
├── .env.example                            # Template de variaveis de ambiente
└── requirements.txt
```

---

## Etapas do notebook

### Fase 1 - Pipeline de ML

| Etapa | Conteudo |
|-------|----------|
| 1 | Carregamento, descricao do problema e metrica prioritaria (Recall) |
| 2 | EDA: distribuicoes, balanceamento (63/37), correlacoes, PCA, outliers |
| 3 | Preprocessamento: limpeza, encoding, split estratificado, StandardScaler |
| 4 | 4 modelos com GridSearchCV + StratifiedKFold otimizando Recall |
| 5 | Avaliacao: Recall, F1, Accuracy, Precision, matriz de confusao, ROC/AUC |
| 6 | Interpretabilidade com SHAP (KernelExplainer - beeswarm + waterfall) |
| 7 | Analise de erros (FN/FP) e comparativo final dos modelos |
| 8 | Serializacao dos modelos via joblib |

### Fase 2 - Extensoes

| Etapa | Conteudo |
|-------|----------|
| 9 | Algoritmo Genetico: SVM e LR, 3 experimentos cada, analise critica de resultados e calibracao de probabilidade |
| 10 | Monitoramento e logging estruturado (FileHandler + StreamHandler) |
| 11 | Integracao LLM multi-provider (Claude / OpenAI) para interpretacao clinica |

---

## Resultados

| Modelo | Recall | F1 | Accuracy | FN | FP |
|--------|--------|----|----------|----|----|
| GridSearchCV LR | **1.0000** | 0.9655 | 0.9649 | **0** | 3 |
| AG LR | 0.9286 | **0.9630** | **0.9737** | 3 | 0 |
| GridSearchCV SVM | 0.9286 | 0.9630 | 0.9737 | 3 | 0 |
| AG SVM | 0.9048 | 0.9500 | 0.9649 | 4 | 0 |

**Metrica prioritaria:** Recall - minimiza falsos negativos (cancer nao detectado).

O GridSearchCV LR com `scoring='recall'` alcancou Recall=1.0 e FN=0 no conjunto de teste.
O AG LR encontrou fitness CV superior (0.9697 vs 0.9588) com objetivo composto, oferecendo
melhor equilibrio geral ao custo de 3 FN. A diferenca reflete objetivos de otimizacao
distintos, nao falha do AG - detalhado na celula 9.8 do notebook.

---

## Como executar

### Prerequisitos

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 1. Notebook (treinamento completo)

```bash
jupyter notebook notebooks/breast-cancer-classification.ipynb
```

Execute todas as celulas em ordem (Kernel > Restart & Run All).
Os modelos serao salvos em `models/` e logs em `logs/`.

### 2. API REST

```bash
uvicorn api.main:app --reload
```

Documentacao interativa: `http://localhost:8000/docs`

| Metodo | Rota | Descricao |
|--------|------|-----------|
| GET | `/health` | Status da API, modelos carregados e LLM |
| POST | `/predict` | Predicao com os 4 modelos simultaneamente |
| POST | `/interpret` | Predicao + interpretacao clinica via LLM |

### 3. Interface web

Com a API rodando em paralelo:

```bash
streamlit run app/streamlit_app.py
```

Acesse `http://localhost:8501`. A interface oferece:
- Formulario com os 30 campos morfologicos e presets rapidos (Benigno, Maligno alto, Maligno extremo)
- Predicao dos 4 modelos com probabilidades
- Interpretacao clinica gerada por LLM com renderizacao markdown
- Historico de predicoes da sessao

---

## Configuracao LLM

```bash
cp .env.example .env
# editar .env com a chave do provider desejado
```

```env
LLM_PROVIDER=claude          # ou "openai"
ANTHROPIC_API_KEY=sk-ant-... # se LLM_PROVIDER=claude
OPENAI_API_KEY=sk-...        # se LLM_PROVIDER=openai
```

Sem chave configurada, `/predict` funciona normalmente; `/interpret` retorna
`interpretacao=""` e `llm_disponivel=false`.

---

## Decisoes tecnicas

| Decisao | Escolha | Justificativa |
|---------|---------|---------------|
| Metrica primaria | Recall | FN (cancer nao detectado) e clinicamente mais grave que FP |
| Balanceamento | StratifiedKFold sem SMOTE | Desbalanceamento moderado 63/37; SMOTE adicionaria ruido sintetico em dataset pequeno |
| Fitness AG | 0.4*Recall + 0.4*F1 + 0.2*Accuracy | Evita degeneracao para "classifica tudo como Maligno" que Recall puro permite |
| Logging | stdlib `logging` dual handler | Zero dependencias extras; arquivo persistente e saida do notebook simultaneamente |
| LLM | Multi-provider Claude/OpenAI | Flexibilidade de custo sem alterar logica do sistema; leitura sempre do .env |
| Calibracao | CalibratedClassifierCV no LR GridSearch | Corrige miscalibracao causada por C=0.01/L1 sem alterar fronteira de decisao |
