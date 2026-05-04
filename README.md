# IA For Devs: Fase 1: Tech Challenge

Classificação binária supervisionada de tumores mamários (FIAP IA Foundations).

## Dataset

**Breast Cancer Wisconsin (Diagnostic)**: 569 amostras, 30 features numéricas extraídas de imagens de aspirados por agulha fina (FNA). Target: `M` (maligno) / `B` (benigno).

Download: [kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)

Salvar o arquivo baixado em `data/data.csv`.

## Estrutura

```
ia-for-devs-phase1-tech-challenge/
├── api/
│   ├── main.py        # FastAPI - endpoint de predição
│   ├── examples.sh    # curls de exemplo
│   └── README.md      # documentação da API
├── data/
│   └── data.csv       # dataset
├── models/
│   ├── breast_cancer_svm.joblib                    # modelo serializado (gerado na Etapa 8)
│   └── breast_cancer_regressao_logistica.joblib    # modelo serializado (gerado na Etapa 8)
├── notebooks/
│   └── breast-cancer-classification.ipynb   # notebook principal
├── README.md
└── requirements.txt
```

## Como executar

```bash
# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Instalar dependências
pip install -r requirements.txt

# Abrir o notebook
jupyter notebook notebooks/breast-cancer-classification.ipynb
```

Executar as células em ordem, de cima para baixo.

## Etapas

1. **Dados e Modelos**: descrição do problema, justificativa de classificação binária supervisionada, métrica prioritária (recall)
2. **EDA**: distribuições, balanceamento, correlação entre features, seleção exploratória
3. **Pré-processamento**: limpeza, encoding, split estratificado, StandardScaler
4. **Modelagem e Tuning**: 4 modelos com `StratifiedKFold` + `GridSearchCV` otimizando recall
5. **Avaliação**: accuracy, recall, F1, precision, matriz de confusão, curva ROC/AUC
6. **Interpretação SHAP**: explicabilidade individual via `KernelExplainer` (beeswarm + waterfall)
7. **Discussão**: análise comparativa SVM × Regressão Logística, trade-off FN/FP, limitações e próximos passos
8. **Serialização**: exportação dos modelos SVM e Regressão Logística via `joblib` para a pasta `models/`

## Extra: API REST de predição

Os modelos treinados estão disponíveis como uma API REST construída com FastAPI. Dado um conjunto de 30 features morfológicas, a API retorna o diagnóstico previsto (Maligno/Benigno) e a probabilidade estimada para os dois modelos: SVM e Regressão Logística.

Requer a execução da Etapa 8 do notebook para gerar os arquivos dos modelos antes de subir a API.

Veja mais detalhes em [`api/README.md`](api/README.md).

## Licença

MIT - veja [LICENSE](LICENSE).
