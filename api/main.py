from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATHS = {
    "svm": MODELS_DIR / "breast_cancer_svm.joblib",
    "regressao_logistica": MODELS_DIR / "breast_cancer_regressao_logistica.joblib",
}
LABEL_MAP = {0: "Benigno", 1: "Maligno"}

# Ordem exata das colunas usada no treino
FEATURE_COLUMNS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

pipelines: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            raise RuntimeError(f"Modelo '{name}' não encontrado em {path}. Execute a Etapa 8 do notebook.")
        pipelines[name] = joblib.load(path)
    yield


app = FastAPI(
    title="Breast Cancer Classifier",
    description="Predição de tumor maligno/benigno a partir de features morfológicas do núcleo celular.",
    version="2.0.0",
    lifespan=lifespan,
)


class TumorFeatures(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float


class ModelPrediction(BaseModel):
    diagnostico: str
    probabilidade_maligno: float


class PredictionResponse(BaseModel):
    svm: ModelPrediction
    regressao_logistica: ModelPrediction


def _predict_single(pipeline, df: pd.DataFrame) -> ModelPrediction:
    pred = int(pipeline.predict(df)[0])
    proba = float(pipeline.predict_proba(df)[0][1])
    return ModelPrediction(diagnostico=LABEL_MAP[pred], probabilidade_maligno=round(proba, 4))


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TumorFeatures):
    data = features.model_dump()
    data["concave points_mean"] = data.pop("concave_points_mean")
    data["concave points_se"] = data.pop("concave_points_se")
    data["concave points_worst"] = data.pop("concave_points_worst")

    df = pd.DataFrame([data])[FEATURE_COLUMNS]

    try:
        return PredictionResponse(
            svm=_predict_single(pipelines["svm"], df),
            regressao_logistica=_predict_single(pipelines["regressao_logistica"], df),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelos": {name: path.name for name, path in MODEL_PATHS.items()},
    }
