"""
Interface web para o Breast Cancer Classifier API.
Execucao: streamlit run app/streamlit_app.py
Requer: API rodando em http://localhost:8000 (uvicorn api.main:app --reload)
"""
import os
import time
from datetime import datetime

import requests
import streamlit as st

# =============================================================================
# Configuracao da pagina
# =============================================================================
st.set_page_config(
    page_title="Classificador de Cancer de Mama",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS customizado
# =============================================================================
st.markdown("""
<style>
    .resultado-maligno {
        background: linear-gradient(135deg, #3d0000, #7a1a1a);
        border-left: 4px solid #C44E52;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .resultado-benigno {
        background: linear-gradient(135deg, #003d00, #1a5c1a);
        border-left: 4px solid #55A868;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .resultado-titulo {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .interpretacao-box {
        background-color: #1a1f2e;
        border-left: 4px solid #4C72B0;
        padding: 20px;
        border-radius: 8px;
        margin-top: 16px;
        line-height: 1.7;
    }
    .badge-llm {
        background-color: #2a3050;
        border: 1px solid #4C72B0;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 12px;
    }
    .metric-card {
        background-color: #1E2130;
        padding: 12px 16px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Valores padrao das features (medias do dataset)
# =============================================================================
FEATURE_DEFAULTS = {
    "radius_mean": 14.13, "texture_mean": 19.29, "perimeter_mean": 91.97,
    "area_mean": 654.89, "smoothness_mean": 0.0964, "compactness_mean": 0.1043,
    "concavity_mean": 0.0888, "concave_points_mean": 0.0489, "symmetry_mean": 0.1812,
    "fractal_dimension_mean": 0.0628, "radius_se": 0.4052, "texture_se": 1.2169,
    "perimeter_se": 2.8661, "area_se": 40.34, "smoothness_se": 0.007041,
    "compactness_se": 0.02548, "concavity_se": 0.03189, "concave_points_se": 0.01180,
    "symmetry_se": 0.02054, "fractal_dimension_se": 0.003795, "radius_worst": 16.27,
    "texture_worst": 25.68, "perimeter_worst": 107.26, "area_worst": 880.58,
    "smoothness_worst": 0.1324, "compactness_worst": 0.2543, "concavity_worst": 0.2722,
    "concave_points_worst": 0.1146, "symmetry_worst": 0.2901, "fractal_dimension_worst": 0.08395,
}

# Exemplos reais do dataset Wisconsin
EXEMPLOS = {
    "benigno": {
        "label": "Benigno tipico",
        "emoji": "✅",
        "help": "Tumor benigno classico - valores morfologicos dentro do padrao normal",
        "valores": {
            "radius_mean": 12.32, "texture_mean": 15.70, "perimeter_mean": 78.92,
            "area_mean": 474.0,   "smoothness_mean": 0.0921, "compactness_mean": 0.0689,
            "concavity_mean": 0.0320, "concave_points_mean": 0.0200, "symmetry_mean": 0.1720,
            "fractal_dimension_mean": 0.0617, "radius_se": 0.2690, "texture_se": 0.9520,
            "perimeter_se": 1.870, "area_se": 18.50, "smoothness_se": 0.006015,
            "compactness_se": 0.01235, "concavity_se": 0.01500, "concave_points_se": 0.00694,
            "symmetry_se": 0.01892, "fractal_dimension_se": 0.003101, "radius_worst": 13.50,
            "texture_worst": 18.20, "perimeter_worst": 86.0, "area_worst": 562.0,
            "smoothness_worst": 0.1180, "compactness_worst": 0.1370, "concavity_worst": 0.0800,
            "concave_points_worst": 0.0498, "symmetry_worst": 0.2480, "fractal_dimension_worst": 0.0748,
        },
    },
    "maligno_alto": {
        "label": "Maligno - alta prob.",
        "emoji": "⚠️",
        "help": "Tumor maligno com probabilidade elevada (~80-90%) - features claramente alteradas",
        "valores": {
            "radius_mean": 17.50, "texture_mean": 21.30, "perimeter_mean": 115.0,
            "area_mean": 955.0,   "smoothness_mean": 0.1050, "compactness_mean": 0.1530,
            "concavity_mean": 0.1550, "concave_points_mean": 0.0850, "symmetry_mean": 0.1950,
            "fractal_dimension_mean": 0.0648, "radius_se": 0.6480, "texture_se": 1.610,
            "perimeter_se": 4.480, "area_se": 68.0, "smoothness_se": 0.009000,
            "compactness_se": 0.03490, "concavity_se": 0.04760, "concave_points_se": 0.01640,
            "symmetry_se": 0.02380, "fractal_dimension_se": 0.005000, "radius_worst": 20.80,
            "texture_worst": 27.50, "perimeter_worst": 137.0, "area_worst": 1355.0,
            "smoothness_worst": 0.1450, "compactness_worst": 0.3540, "concavity_worst": 0.3800,
            "concave_points_worst": 0.1650, "symmetry_worst": 0.3180, "fractal_dimension_worst": 0.0949,
        },
    },
    "maligno_extremo": {
        "label": "Maligno - extremo",
        "emoji": "🚨",
        "help": "Tumor maligno agressivo (~99%) - valores morfologicos severamente alterados",
        "valores": {
            "radius_mean": 24.63, "texture_mean": 29.89, "perimeter_mean": 163.7,
            "area_mean": 1844.0,  "smoothness_mean": 0.1265, "compactness_mean": 0.3035,
            "concavity_mean": 0.3515, "concave_points_mean": 0.1628, "symmetry_mean": 0.2349,
            "fractal_dimension_mean": 0.0749, "radius_se": 1.471, "texture_se": 2.930,
            "perimeter_se": 10.50, "area_se": 196.0, "smoothness_se": 0.009635,
            "compactness_se": 0.06483, "concavity_se": 0.09564, "concave_points_se": 0.02735,
            "symmetry_se": 0.03866, "fractal_dimension_se": 0.007351, "radius_worst": 32.01,
            "texture_worst": 40.72, "perimeter_worst": 214.9, "area_worst": 3432.0,
            "smoothness_worst": 0.1844, "compactness_worst": 0.7448, "concavity_worst": 0.9317,
            "concave_points_worst": 0.3215, "symmetry_worst": 0.5354, "fractal_dimension_worst": 0.1407,
        },
    },
}

# =============================================================================
# Estado da sessao
# =============================================================================
if "historico" not in st.session_state:
    st.session_state["historico"] = []
if "ultimo_input" not in st.session_state:
    st.session_state["ultimo_input"] = {}

# Inicializa os campos dos formularios com os valores default (somente na primeira carga)
for _prefix in ("pred", "interp"):
    for _feat, _val in FEATURE_DEFAULTS.items():
        _key = f"{_prefix}_{_feat}"
        if _key not in st.session_state:
            st.session_state[_key] = float(_val)

# =============================================================================
# Sidebar
# =============================================================================
api_url = os.environ.get("API_URL", "http://localhost:8000")

with st.sidebar:
    st.subheader("Status da API")
    if st.button("Verificar saude da API", width="stretch"):
        try:
            resp = requests.get(f"{api_url}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                st.success("API online")
                st.write("**Modelos disponiveis:**")
                for m in data.get("modelos_disponiveis", []):
                    st.write(f"  - `{m}`")
                llm = data.get("llm", {})
                if llm.get("disponivel"):
                    st.success(f"LLM: {llm['provider']}")
                else:
                    st.warning("LLM: indisponivel (sem chave de API)")
            else:
                st.error(f"Erro HTTP {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("API nao acessivel. Verifique se esta rodando.")
        except Exception as e:
            st.error(f"Erro: {e}")

    st.divider()
    st.subheader("Sobre")
    st.markdown("""
    Este sistema classifica tumores de mama como **Maligno** ou **Benigno**
    com base em 30 features morfologicas extraidas por aspiracao por agulha fina (PAAF).

    **Modelos disponibilizados:**
    - SVM (GridSearchCV + AG)
    - Regressao Logistica (GridSearchCV + AG)

    **Dados:** Breast Cancer Wisconsin (569 amostras)

    **Versao:** 2.0 - Fase 2 Tech Challenge FIAP
    """)


def _aplicar_preset(prefix: str, valores: dict):
    """Preenche os campos do formulario com valores do preset via session_state."""
    for feat, val in valores.items():
        st.session_state[f"{prefix}_{feat}"] = float(val)


def _render_preset_buttons(prefix: str):
    """Renderiza botoes de exemplos rapidos e botao de reset."""
    st.markdown("**Exemplos rapidos:**")
    cols = st.columns([1, 1, 1, 1])
    presets = list(EXEMPLOS.items())
    for i, (key, ex) in enumerate(presets):
        with cols[i]:
            if st.button(
                f"{ex['emoji']} {ex['label']}",
                key=f"btn_{prefix}_{key}",
                help=ex["help"],
                width="stretch",
            ):
                _aplicar_preset(prefix, ex["valores"])
                st.rerun()
    with cols[3]:
        if st.button(
            "↺ Usar medias",
            key=f"btn_{prefix}_reset",
            help="Reseta todos os campos para a media do dataset de treino",
            width="stretch",
        ):
            _aplicar_preset(prefix, FEATURE_DEFAULTS)
            st.rerun()
    st.divider()


FEATURE_STEPS = {k: 0.0001 if v < 0.1 else (0.01 if v < 1 else (0.1 if v < 10 else 1.0))
                 for k, v in FEATURE_DEFAULTS.items()}

GROUPS = {
    "Media (Mean)": [k for k in FEATURE_DEFAULTS if k.endswith("_mean")],
    "Erro padrao (SE)":   [k for k in FEATURE_DEFAULTS if k.endswith("_se")],
    "Pior valor (Worst)": [k for k in FEATURE_DEFAULTS if k.endswith("_worst")],
}

NOMES_PT = {
    "radius": "Raio", "texture": "Textura", "perimeter": "Perimetro",
    "area": "Area", "smoothness": "Suavidade", "compactness": "Compacidade",
    "concavity": "Concavidade", "concave_points": "Pontos concavos",
    "symmetry": "Simetria", "fractal_dimension": "Dimensao fractal",
}

def label_pt(feature_key: str) -> str:
    base = "_".join(feature_key.split("_")[:-1])
    return NOMES_PT.get(base, base.replace("_", " ").title())


def render_feature_form(prefix: str) -> dict:
    """Renderiza formulario com os 30 campos agrupados. Retorna dict de valores."""
    valores = {}
    for grupo, features in GROUPS.items():
        with st.expander(f"**{grupo}** ({len(features)} features)", expanded=(grupo == "Media (Mean)")):
            cols = st.columns(2)
            for idx, feat in enumerate(features):
                with cols[idx % 2]:
                    valores[feat] = st.number_input(
                        label=f"{label_pt(feat)}",
                        step=FEATURE_STEPS[feat],
                        format="%.4f",
                        key=f"{prefix}_{feat}",
                    )
    return valores


def card_resultado(titulo: str, diag: str, proba: float):
    css_class = "resultado-maligno" if diag == "Maligno" else "resultado-benigno"
    emoji = "⚠️" if diag == "Maligno" else "✅"
    st.markdown(
        f'<div class="{css_class}">'
        f'<div class="resultado-titulo">{emoji} {titulo}: {diag}</div>'
        f'Probabilidade de malignidade: <strong>{proba:.1%}</strong>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(proba)


# =============================================================================
# Titulo principal
# =============================================================================
st.title("🔬 Classificador de Cancer de Mama")
st.caption("Sistema de apoio a decisao clinica - Breast Cancer Wisconsin Dataset")

# =============================================================================
# Abas
# =============================================================================
aba1, aba2, aba3 = st.tabs(["Predicao", "Interpretar com LLM", "Historico"])

# ─────────────────────────────────────────────────────────────────────────────
# ABA 1: PREDICAO
# ─────────────────────────────────────────────────────────────────────────────
with aba1:
    st.subheader("Classificar tumor")
    st.info(
        "Preencha os valores das features morfologicas do tumor. "
        "Os valores padrao sao as medias do dataset de treino.",
        icon="ℹ️",
    )

    _render_preset_buttons("pred")
    valores_pred = render_feature_form("pred")

    col_btn, col_modelo = st.columns([2, 1])
    with col_modelo:
        opcoes_modelo = ["svm", "regressao_logistica", "svm_ag", "regressao_logistica_ag"]
        modelos_selecionados = st.multiselect(
            "Modelos a comparar",
            options=opcoes_modelo,
            default=opcoes_modelo,
        )
    with col_btn:
        st.write("")
        st.write("")
        btn_predict = st.button("🔍 Classificar Tumor", type="primary", width="stretch")

    if btn_predict:
        if not modelos_selecionados:
            st.warning("Selecione ao menos um modelo.")
        else:
            payload = valores_pred.copy()
            try:
                with st.spinner("Consultando API..."):
                    resp = requests.post(f"{api_url}/predict", json=payload, timeout=30)

                if resp.status_code == 200:
                    data = resp.json()
                    st.divider()
                    st.subheader("Resultados")
                    cols = st.columns(len(modelos_selecionados))
                    for i, modelo_key in enumerate(modelos_selecionados):
                        nome_exibicao = modelo_key.replace("_", " ").upper()
                        r = data.get(modelo_key)
                        with cols[i]:
                            if r:
                                card_resultado(nome_exibicao, r["diagnostico"], r["probabilidade_maligno"])
                            else:
                                st.warning(
                                    f"**{nome_exibicao}** nao disponivel.  \n"
                                    "Execute a Etapa 9 do notebook para treinar o modelo AG."
                                )

                    # Registrar no historico
                    primeiro_modelo = modelos_selecionados[0]
                    if primeiro_modelo in data:
                        r0 = data[primeiro_modelo]
                        st.session_state["historico"].append({
                            "Timestamp":           datetime.now().strftime("%H:%M:%S"),
                            "Modelo":              primeiro_modelo,
                            "Diagnostico":         r0["diagnostico"],
                            "Prob. Maligno":       f'{r0["probabilidade_maligno"]:.1%}',
                        })
                    st.session_state["ultimo_input"] = valores_pred

                else:
                    st.error(f"Erro da API: {resp.status_code} - {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error("Nao foi possivel conectar a API. Verifique se esta rodando.")
            except Exception as e:
                st.error(f"Erro: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ABA 2: INTERPRETAR COM LLM
# ─────────────────────────────────────────────────────────────────────────────
with aba2:
    st.subheader("Interpretacao clinica via LLM")
    st.info(
        "O LLM gera uma explicacao em linguagem natural do diagnostico, "
        "adequada para medicos sem formacao em ML. "
        "Requer chave de API configurada no servidor.",
        icon="🤖",
    )

    usar_ultimo = False
    if st.session_state.get("ultimo_input"):
        usar_ultimo = st.checkbox("Usar os mesmos valores da aba Predicao", value=True)

    if usar_ultimo and st.session_state.get("ultimo_input"):
        valores_interp = st.session_state["ultimo_input"]
        st.success("Usando valores da ultima predicao.")
    else:
        st.markdown("**Preencha as features:**")
        _render_preset_buttons("interp")
        valores_interp = render_feature_form("interp")

    col_m, col_h = st.columns([1, 2])
    with col_m:
        modelo_interp = st.selectbox(
            "Modelo para interpretacao",
            options=["svm", "regressao_logistica", "svm_ag", "regressao_logistica_ag"],
            index=0,
        )
    with col_h:
        historico_clinico = st.text_area(
            "Historico clinico do paciente (opcional)",
            placeholder="Ex: Paciente feminina, 55 anos. Historico familiar de cancer de mama...",
            height=100,
        )

    btn_interpret = st.button("🤖 Gerar Interpretacao Clinica", type="primary", width="stretch")

    if btn_interpret:
        payload_interp = {
            "features":         valores_interp,
            "modelo":           modelo_interp,
            "historico_clinico": historico_clinico,
        }
        try:
            with st.spinner("Consultando modelo e LLM... (pode levar alguns segundos)"):
                t0   = time.time()
                resp = requests.post(f"{api_url}/interpret", json=payload_interp, timeout=60)
                t1   = time.time()

            if resp.status_code == 200:
                data = resp.json()
                st.divider()

                diag  = data["diagnostico"]
                proba = data["probabilidade_maligno"]
                card_resultado(modelo_interp.upper(), diag, proba)

                if data["llm_disponivel"] and data["interpretacao"]:
                    st.markdown(
                        f'<div class="badge-llm">🤖 Gerado por: {data["llm_provider"]}</div>',
                        unsafe_allow_html=True,
                    )
                    with st.container(border=True):
                        st.caption(f"Relatorio gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
                        st.markdown(data["interpretacao"])
                    st.caption(f"Tempo de resposta: {t1-t0:.1f}s")
                else:
                    st.warning(
                        "LLM nao disponivel no servidor. "
                        "Configure LLM_PROVIDER e a chave de API (ANTHROPIC_API_KEY ou OPENAI_API_KEY)."
                    )

                st.session_state["historico"].append({
                    "Timestamp":     datetime.now().strftime("%H:%M:%S"),
                    "Modelo":        modelo_interp,
                    "Diagnostico":   diag,
                    "Prob. Maligno": f"{proba:.1%}",
                })

            elif resp.status_code == 400:
                st.error(f"Modelo nao disponivel: {resp.json().get('detail', resp.text)}")
            else:
                st.error(f"Erro da API: {resp.status_code} - {resp.text}")

        except requests.exceptions.ConnectionError:
            st.error("Nao foi possivel conectar a API.")
        except Exception as e:
            st.error(f"Erro: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ABA 3: HISTORICO
# ─────────────────────────────────────────────────────────────────────────────
with aba3:
    st.subheader("Historico de predicoes da sessao")

    if not st.session_state["historico"]:
        st.info("Nenhuma predicao realizada nesta sessao.")
    else:
        import pandas as pd
        df_hist = pd.DataFrame(st.session_state["historico"])

        col_info, col_btn_limpar = st.columns([3, 1])
        with col_info:
            st.write(f"**{len(df_hist)} predicao(oes)** nesta sessao")
        with col_btn_limpar:
            if st.button("🗑️ Limpar historico", width="stretch"):
                st.session_state["historico"] = []
                st.rerun()

        def colorir_diagnostico(val):
            if val == "Maligno":
                return "background-color: #7a1a1a; color: white"
            elif val == "Benigno":
                return "background-color: #1a5c1a; color: white"
            return ""

        st.dataframe(
            df_hist.style.map(colorir_diagnostico, subset=["Diagnostico"]),
            use_container_width=True,
            hide_index=True,
        )

        maligno_count = (df_hist["Diagnostico"] == "Maligno").sum()
        benigno_count = (df_hist["Diagnostico"] == "Benigno").sum()

        col_m, col_b, col_t = st.columns(3)
        with col_m:
            st.metric("Malignos", maligno_count)
        with col_b:
            st.metric("Benignos", benigno_count)
        with col_t:
            taxa = maligno_count / len(df_hist) * 100 if df_hist.shape[0] > 0 else 0
            st.metric("Taxa de malignidade", f"{taxa:.0f}%")
