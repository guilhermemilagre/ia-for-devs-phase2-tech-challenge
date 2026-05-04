# Breast Cancer Classifier API

API REST para predição de tumores mamários (maligno/benigno) a partir de features morfológicas do núcleo celular. Retorna a predição dos dois modelos treinados no notebook: SVM e Regressão Logística.

## Pré-requisito

Executar a **Etapa 8** do notebook para gerar os arquivos abaixo antes de subir a API:

- `models/breast_cancer_svm.joblib`
- `models/breast_cancer_regressao_logistica.joblib`

## Instalação

```bash
pip install -r requirements.txt
```

## Como rodar

Execute a partir da **raiz do projeto** (não de dentro da pasta `api/`):

```bash
uvicorn api.main:app --reload
```

API disponível em `http://127.0.0.1:8000`.
Documentação interativa (Swagger): `http://127.0.0.1:8000/docs`.

## Endpoints

### `GET /health`
Verifica se a API está no ar e quais modelos estão carregados.

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "modelos": {
    "svm": "breast_cancer_svm.joblib",
    "regressao_logistica": "breast_cancer_regressao_logistica.joblib"
  }
}
```

---

### `POST /predict`
Recebe as 30 features morfológicas e retorna o diagnóstico previsto pelos dois modelos.

**Payload**: os 30 campos numéricos do dataset Wisconsin (mesmo formato do CSV, com `_` no lugar de espaço).

**Response**:

```json
{
  "svm": {
    "diagnostico": "Maligno",
    "probabilidade_maligno": 0.9453
  },
  "regressao_logistica": {
    "diagnostico": "Maligno",
    "probabilidade_maligno": 0.8821
  }
}
```

- `diagnostico`: `"Maligno"` ou `"Benigno"`
- `probabilidade_maligno`: probabilidade estimada pelo modelo para a classe maligna (0 a 1)

## Exemplos prontos

```bash
./api/examples.sh
```

O script executa o health check e 3 casos reais do CSV (2 malignos, 1 benigno).

## Campos do payload

| Campo | Tipo |
|---|---|
| `radius_mean` | float |
| `texture_mean` | float |
| `perimeter_mean` | float |
| `area_mean` | float |
| `smoothness_mean` | float |
| `compactness_mean` | float |
| `concavity_mean` | float |
| `concave_points_mean` | float |
| `symmetry_mean` | float |
| `fractal_dimension_mean` | float |
| `radius_se` | float |
| `texture_se` | float |
| `perimeter_se` | float |
| `area_se` | float |
| `smoothness_se` | float |
| `compactness_se` | float |
| `concavity_se` | float |
| `concave_points_se` | float |
| `symmetry_se` | float |
| `fractal_dimension_se` | float |
| `radius_worst` | float |
| `texture_worst` | float |
| `perimeter_worst` | float |
| `area_worst` | float |
| `smoothness_worst` | float |
| `compactness_worst` | float |
| `concavity_worst` | float |
| `concave_points_worst` | float |
| `symmetry_worst` | float |
| `fractal_dimension_worst` | float |
