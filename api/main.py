from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values

_ENV_PATH = Path(__file__).parent.parent / ".env"
_ENV = dotenv_values(_ENV_PATH)

def _cfg(key: str, default: str = "") -> str:
    """Le sempre do .env; nunca depende de variaveis de ambiente do shell."""
    return _ENV.get(key) or default

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATHS = {
    "svm": MODELS_DIR / "breast_cancer_svm.joblib",
    "regressao_logistica": MODELS_DIR / "breast_cancer_regressao_logistica.joblib",
    "svm_ag": MODELS_DIR / "breast_cancer_svm_ag.joblib",
    "regressao_logistica_ag": MODELS_DIR / "breast_cancer_regressao_logistica_ag.joblib",
    "regressao_logistica_calibrado": MODELS_DIR / "breast_cancer_regressao_logistica_calibrado.joblib",
}

FEATURE_COLUMNS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

# Medias do dataset de treino (da Etapa 2 do notebook) para identificar features extremas
FEATURE_MEANS = {
    "radius_mean": 14.13, "texture_mean": 19.29, "perimeter_mean": 91.97,
    "area_mean": 654.89, "smoothness_mean": 0.0964, "compactness_mean": 0.1043,
    "concavity_mean": 0.0888, "concave points_mean": 0.0489, "symmetry_mean": 0.1812,
    "fractal_dimension_mean": 0.0628, "radius_se": 0.4052, "texture_se": 1.2169,
    "perimeter_se": 2.8661, "area_se": 40.34, "smoothness_se": 0.007041,
    "compactness_se": 0.02548, "concavity_se": 0.03189, "concave points_se": 0.01180,
    "symmetry_se": 0.02054, "fractal_dimension_se": 0.003795, "radius_worst": 16.27,
    "texture_worst": 25.68, "perimeter_worst": 107.26, "area_worst": 880.58,
    "smoothness_worst": 0.1324, "compactness_worst": 0.2543, "concavity_worst": 0.2722,
    "concave points_worst": 0.1146, "symmetry_worst": 0.2901, "fractal_dimension_worst": 0.08395,
}

LABEL_MAP = {0: "Benigno", 1: "Maligno"}

pipelines: dict = {}
llm_client = None
llm_provider_str = "indisponivel"
LLM_ATIVO = False

SYSTEM_PROMPT_DIAGNOSTICO = (
    "Voce e um assistente de apoio clinico especializado em oncologia mamaria. "
    "Sua funcao e transformar resultados de modelos de aprendizado de maquina em linguagem "
    "natural clara e acessivel para medicos oncologistas.\n\n"
    "Regras obrigatorias:\n"
    "1. Sempre escreva em portugues brasileiro formal.\n"
    "2. Nunca emita um diagnostico definitivo: seus outputs sao apoio a decisao.\n"
    "3. Sempre inclua a recomendacao de confirmacao clinica.\n"
    "4. Use terminologia medica precisa, mas explique termos tecnicos de ML.\n"
    "5. Seja objetivo: 3-5 frases por secao, sem redundancias.\n"
    "6. Estruture a resposta exatamente nas secoes solicitadas."
)


def _setup_llm():
    global llm_client, llm_provider_str, LLM_ATIVO
    provider = _cfg("LLM_PROVIDER", "claude").lower()

    if provider == "claude":
        try:
            import anthropic
            api_key = _cfg("ANTHROPIC_API_KEY")
            if api_key:
                llm_client       = anthropic.Anthropic(api_key=api_key)
                llm_provider_str = f"claude ({_cfg('LLM_MODEL', 'claude-haiku-4-5')})"
                LLM_ATIVO        = True
        except ImportError:
            pass

    elif provider == "openai":
        try:
            import openai as oai
            api_key = _cfg("OPENAI_API_KEY")
            if api_key:
                llm_client       = oai.OpenAI(api_key=api_key)
                llm_provider_str = f"openai ({_cfg('LLM_MODEL', 'gpt-4o-mini')})"
                LLM_ATIVO        = True
        except ImportError:
            pass


def _chamar_llm(prompt_sistema: str, prompt_usuario: str,
                max_tokens: int = 800, temperatura: float = 0.3) -> str:
    if not LLM_ATIVO or llm_client is None:
        return ""
    provider = _cfg("LLM_PROVIDER", "claude").lower()
    try:
        if provider == "claude":
            import anthropic
            modelo = _cfg("LLM_MODEL", "claude-haiku-4-5")
            resp = llm_client.messages.create(
                model=modelo, max_tokens=max_tokens, temperature=temperatura,
                system=prompt_sistema,
                messages=[{"role": "user", "content": prompt_usuario}],
            )
            return resp.content[0].text
        elif provider == "openai":
            modelo = _cfg("LLM_MODEL", "gpt-4o-mini")
            resp = llm_client.chat.completions.create(
                model=modelo, max_tokens=max_tokens, temperature=temperatura,
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user",   "content": prompt_usuario},
                ],
            )
            return resp.choices[0].message.content
    except Exception:
        return ""
    return ""


def _top_features_extremas(df: pd.DataFrame, n: int = 5) -> list[dict]:
    row = df.iloc[0]
    desvios = []
    for col in FEATURE_COLUMNS:
        col_original = col.replace("concave points_", "concave_points_")
        val  = float(row[col])
        mean = FEATURE_MEANS.get(col, 0.0)
        desvio = abs(val - mean) / (mean + 1e-9)
        desvios.append({"feature": col, "valor": val, "media": mean, "desvio_rel": desvio})
    desvios.sort(key=lambda x: x["desvio_rel"], reverse=True)
    return desvios[:n]


def _gerar_prompt_interpretacao(diagnostico: str, probabilidade: float,
                                 top_features: list[dict], historico_clinico: str) -> str:
    features_str = "\n".join([
        f"   - {f['feature']}: valor={f['valor']:.4f} (media dataset={f['media']:.4f}, "
        f"desvio relativo={f['desvio_rel']:.1%})"
        for f in top_features
    ])
    hist_section = (
        f"\nHISTORICO CLINICO (dados do prontuario):\n{historico_clinico}\n"
        if historico_clinico.strip() else ""
    )
    return (
        f"Analise os resultados do modelo de classificacao de cancer de mama:\n\n"
        f"DIAGNOSTICO DO MODELO: {diagnostico}\n"
        f"PROBABILIDADE DE MALIGNIDADE: {probabilidade:.1%}\n"
        f"PROBABILIDADE DE BENIGNIDADE: {1-probabilidade:.1%}\n\n"
        f"TOP 5 FEATURES COM MAIOR DESVIO DA MEDIA DO DATASET:\n{features_str}\n"
        f"{hist_section}\n"
        f"Gere uma interpretacao clinica estruturada com as seguintes secoes:\n\n"
        f"## Sintese do Diagnostico\n"
        f"(1-2 frases: diagnostico do modelo, probabilidade, nivel de confianca)\n\n"
        f"## Fatores Mais Relevantes\n"
        f"(Explique em linguagem clinica quais features mais se desviam do padrao esperado)\n\n"
        f"## Recomendacao de Acompanhamento\n"
        f"(O que o medico deve considerar baseado neste resultado)\n\n"
        f"## Limitacoes e Responsabilidade\n"
        f"(Lembrete obrigatorio sobre o papel do modelo como ferramenta de apoio)"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name, path in MODEL_PATHS.items():
        if path.exists():
            pipelines[name] = joblib.load(path)
        elif name in ("svm", "regressao_logistica"):
            raise RuntimeError(
                f"Modelo '{name}' nao encontrado em {path}. Execute a Etapa 8 do notebook."
            )
    _setup_llm()
    yield


app = FastAPI(
    title="Breast Cancer Classifier",
    description=(
        "Predicao de tumor maligno/benigno a partir de features morfologicas do nucleo celular. "
        "Inclui endpoint de interpretacao via LLM para geracao de laudos em linguagem natural."
    ),
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
    svm_ag: Optional[ModelPrediction] = None
    regressao_logistica_ag: Optional[ModelPrediction] = None
    regressao_logistica_calibrado: Optional[ModelPrediction] = None


class InterpretarRequest(BaseModel):
    features: TumorFeatures
    modelo: str = "svm"
    historico_clinico: str = ""


class InterpretacaoResponse(BaseModel):
    modelo: str
    diagnostico: str
    probabilidade_maligno: float
    interpretacao: str
    llm_provider: str
    llm_disponivel: bool


def _features_to_df(features: TumorFeatures) -> pd.DataFrame:
    data = features.model_dump()
    data["concave points_mean"]  = data.pop("concave_points_mean")
    data["concave points_se"]    = data.pop("concave_points_se")
    data["concave points_worst"] = data.pop("concave_points_worst")
    return pd.DataFrame([data])[FEATURE_COLUMNS]


def _predict_single(pipeline, df: pd.DataFrame) -> ModelPrediction:
    pred  = int(pipeline.predict(df)[0])
    proba = float(pipeline.predict_proba(df)[0][1])
    return ModelPrediction(diagnostico=LABEL_MAP[pred], probabilidade_maligno=round(proba, 4))


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TumorFeatures):
    df = _features_to_df(features)
    try:
        return PredictionResponse(
            svm=_predict_single(pipelines["svm"], df),
            regressao_logistica=_predict_single(pipelines["regressao_logistica"], df),
            svm_ag=(
                _predict_single(pipelines["svm_ag"], df)
                if "svm_ag" in pipelines else None
            ),
            regressao_logistica_ag=(
                _predict_single(pipelines["regressao_logistica_ag"], df)
                if "regressao_logistica_ag" in pipelines else None
            ),
            regressao_logistica_calibrado=(
                _predict_single(pipelines["regressao_logistica_calibrado"], df)
                if "regressao_logistica_calibrado" in pipelines else None
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interpret", response_model=InterpretacaoResponse)
def interpret(request: InterpretarRequest):
    modelo_key = request.modelo.lower().replace(" ", "_").replace("-", "_")
    if modelo_key not in pipelines:
        opcoes = list(pipelines.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Modelo '{request.modelo}' nao disponivel. Opcoes: {opcoes}",
        )

    df = _features_to_df(request.features)
    try:
        pred  = int(pipelines[modelo_key].predict(df)[0])
        proba = float(pipelines[modelo_key].predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predicao: {e}")

    diagnostico = LABEL_MAP[pred]
    top_features = _top_features_extremas(df)

    interpretacao = ""
    if LLM_ATIVO:
        prompt = _gerar_prompt_interpretacao(
            diagnostico=diagnostico,
            probabilidade=proba,
            top_features=top_features,
            historico_clinico=request.historico_clinico,
        )
        interpretacao = _chamar_llm(SYSTEM_PROMPT_DIAGNOSTICO, prompt)

    return InterpretacaoResponse(
        modelo=modelo_key,
        diagnostico=diagnostico,
        probabilidade_maligno=round(proba, 4),
        interpretacao=interpretacao,
        llm_provider=llm_provider_str,
        llm_disponivel=LLM_ATIVO,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelos_disponiveis": list(pipelines.keys()),
        "llm": {
            "disponivel": LLM_ATIVO,
            "provider":   llm_provider_str,
        },
    }
