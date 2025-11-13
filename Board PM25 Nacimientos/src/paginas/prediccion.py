import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from joblib import load
import re

# Define the expected feature columns
REQUIRED_COLUMNS = ['codmunre', 'sexo', 'edad_madre', 'pm25_promedio_gestacion']
MODEL_FILENAME = "prematuro_model.pkl"
JOBLIB_FILENAME = "prematuro_model.joblib"

# Mapa nombre -> código (puedes editar)
MUNICIPIOS_MAP = {
    "Medellín": "05001",
    "Caldas": "05031",
    "Copacabana": "05033",
    "Girardota": "05079",
    "Bello": "05088",
    "Itagüí": "05097",
    "Envigado": "05104",
    "Sabaneta": "05109",
    "La Estrella": "05113",
    "Barbosa": "05011"
}
# Lista de nombres para mostrar en selectbox
MUNICIPIOS = list(MUNICIPIOS_MAP.keys())
# Código -> nombre (reverse map) para normalizar CSV con códigos
CODE_TO_NAME = {v: k for k, v in MUNICIPIOS_MAP.items()}

# --- Feature Engineering / Preprocessing ---
def _normalize_municipio(val):
    """Devuelve el nombre del municipio (como string) esperado por el modelo.
       Acepta: nombre, '05001', '05001 - Medellín' y normaliza."""
    if pd.isna(val):
        return val
    if isinstance(val, (int, float)):
        val = str(int(val))
    val_str = str(val).strip()
    # si contiene guion, tomar la parte derecha o izquierda según formato
    if "-" in val_str:
        left = val_str.split("-")[0].strip()
        right = val_str.split("-")[1].strip()
        # si left es dígitos, mapear a nombre
        digits = re.sub(r"\D", "", left)
        if digits and digits in CODE_TO_NAME:
            return CODE_TO_NAME[digits]
        # si right coincide con un nombre conocido
        if right in MUNICIPIOS_MAP:
            return right
    # si es sólo dígitos, mapear código->nombre
    digits = re.sub(r"\D", "", val_str)
    if digits and digits in CODE_TO_NAME:
        return CODE_TO_NAME[digits]
    # intentar emparejar por nombre ignorando mayúsculas/acentos básicos
    val_norm = val_str.lower()
    for name in MUNICIPIOS:
        if val_norm == name.lower():
            return name
    # si no se puede normalizar, devolver original (modelo puede aceptar otras variantes)
    return val_str

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess dataframe:
    - Normaliza 'codmunre' a nombre de municipio (string)
    - Codifica 'sexo' a 1/0
    - Convierte edad y pm25 a numérico
    """
    df_processed = df.copy()

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_processed.columns]
    if missing_cols:
        st.error(f"Faltan columnas: {missing_cols}")
        raise ValueError("Missing required columns")

    # Sexo -> 1/0
    df_processed['sexo'] = df_processed['sexo'].astype(str).str.upper().str.strip()
    df_processed['sexo'] = df_processed['sexo'].replace(['MASCULINO', 'M'], 1).replace(['FEMENINO', 'F'], 0)

    # Codmunre -> normalizar a NOMBRE de municipio
    df_processed['codmunre'] = df_processed['codmunre'].apply(_normalize_municipio)

    # Edad y pm25 -> numeric
    df_processed['edad_madre'] = pd.to_numeric(df_processed['edad_madre'], errors='coerce')
    df_processed['pm25_promedio_gestacion'] = pd.to_numeric(df_processed['pm25_promedio_gestacion'], errors='coerce')

    # Asegurar columnas en el orden esperado por el modelo
    result = df_processed[REQUIRED_COLUMNS].dropna()
    return result

# --- Mock Model ---
class MockModel:
    def predict_proba(self, X: pd.DataFrame):
        prob_risk = 0.5 + (X['pm25_promedio_gestacion'] / 100) * 0.2
        prob_risk += ((X['edad_madre'] - 27).abs() / 15) * 0.1
        prob_risk = prob_risk.clip(0.1, 0.9)
        return pd.DataFrame({'prob_0': 1 - prob_risk, 'prob_1': prob_risk}).values

    def predict(self, X: pd.DataFrame):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

# --- Model loading ---
def _base_path():
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path(".").resolve()

def load_prediction_model():
    base = _base_path()
    model_path_pkl = base / "models" / MODEL_FILENAME
    model_path_joblib = base / "models" / JOBLIB_FILENAME
    model = None
    if model_path_pkl.exists() or model_path_joblib.exists():
        model_path = model_path_pkl if model_path_pkl.exists() else model_path_joblib
        try:
            if model_path.suffix == ".pkl":
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                model = load(model_path)
            st.success("Modelo cargado correctamente.")
            return model
        except Exception as e:
            st.error(f"Error cargando el modelo desde {model_path}. Causa: {e}")
    st.warning("No se encontró modelo válido. Se usará un modelo simulado (mock).")
    return MockModel()

# --- Página de Streamlit ---
def cargar_prediccion():
    st.set_page_config(page_title="Predicción Prematuro", layout="wide")
    st.title("👶 Predicción: Probabilidad de Nacimiento Prematuro")
    st.markdown("Modelo espera `codmunre` como NOMBRE del municipio (ej. 'Medellín'). Columnas: `codmunre`, `sexo`, `edad_madre`, `pm25_promedio_gestacion`")

    model = load_prediction_model()
    if model is None:
        return

    st.markdown("---")
    st.markdown("### 📊 Opción A: Subir CSV para predicción múltiple")
    uploaded = st.file_uploader("Sube un CSV con las columnas: codmunre, sexo, edad_madre, pm25_promedio_gestacion", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write(f"Vista previa ({df.shape[0]} filas):")
            st.dataframe(df.head())

            if not all(col in df.columns for col in REQUIRED_COLUMNS):
                st.error(f"El CSV debe contener: {REQUIRED_COLUMNS}. Columnas encontradas: {list(df.columns)}")
            else:
                df_to_predict = preprocess_df(df)
                if df_to_predict.empty:
                    st.warning("Preprocesamiento vacío (valores faltantes).")
                    return

                with st.spinner("Realizando predicciones..."):
                    if hasattr(model, "predict_proba"):
                        probs = np.asarray(model.predict_proba(df_to_predict))
                        if hasattr(model, "classes_"):
                            classes = np.asarray(model.classes_)
                            # intentar encontrar índice de la clase positiva (1 o '1' o 'Si' según entrenamiento)
                            if 1 in classes:
                                idx1 = int(np.where(classes == 1)[0][0])
                                idx0 = int(np.where(classes != 1)[0][0])
                            else:
                                idx1 = 1 if probs.shape[1] > 1 else 0
                                idx0 = 0
                        else:
                            idx1 = 1 if probs.shape[1] > 1 else 0
                            idx0 = 0

                        df["prob_no_prematuro"] = pd.Series(probs[:, idx0], index=df_to_predict.index).round(4)
                        df["prob_prematuro"] = pd.Series(probs[:, idx1], index=df_to_predict.index).round(4)

                    df["pred_prematuro"] = pd.Series(model.predict(df_to_predict), index=df_to_predict.index).astype(int)

                st.markdown("#### ✅ Resultados")
                st.dataframe(df.head(200))

                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                st.download_button("Descargar resultados (CSV)", convert_df_to_csv(df), "predicciones_prematuro.csv", "text/csv")
        except Exception as e:
            st.error(f"Error procesando el CSV: {e}")

    st.markdown("---")
    st.markdown("### ✍️ Opción B: Predicción manual (una fila)")
    with st.form("manual_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            municipio_sel = st.selectbox("Municipio", options=MUNICIPIOS, index=0)
        with col2:
            sexo_input = st.selectbox("Sexo", options=["M", "F"], index=0)
        col3, col4 = st.columns(2)
        with col3:
            edad_madre = st.number_input("Edad madre (años)", min_value=10, max_value=60, value=27, step=1)
        with col4:
            pm25_promedio = st.number_input("PM2.5 promedio gestación (µg/m³)", min_value=0.0, value=25.0, format="%.2f", step=0.1)

        submitted = st.form_submit_button("Predecir")
        if submitted:
            input_data = pd.DataFrame([{
                "codmunre": municipio_sel,
                "sexo": sexo_input,
                "edad_madre": edad_madre,
                "pm25_promedio_gestacion": pm25_promedio
            }])
            st.write("Entrada original:")
            st.dataframe(input_data)

            try:
                input_df_processed = preprocess_df(input_data)
                st.write("Entrada preprocesada:")
                st.dataframe(input_df_processed)

                if hasattr(model, "predict_proba"):
                    probs = np.asarray(model.predict_proba(input_df_processed))
                    if hasattr(model, "classes_"):
                        classes = np.asarray(model.classes_)
                        if 1 in classes:
                            idx1 = int(np.where(classes == 1)[0][0])
                            idx0 = int(np.where(classes != 1)[0][0])
                        else:
                            idx1 = 1 if probs.shape[1] > 1 else 0
                            idx0 = 0
                    else:
                        idx1 = 1 if probs.shape[1] > 1 else 0
                        idx0 = 0

                    prob_no = float(probs[0, idx0])
                    prob_si = float(probs[0, idx1])
                    pred = int(model.predict(input_df_processed)[0])
                    result_text = 'Prematuro' if pred == 1 else 'No prematuro'
                    st.success(f"Resultado: {result_text}")
                    st.info(f"Probabilidad No prematuro: {prob_no:.3f} — Probabilidad Prematuro: {prob_si:.3f}")
                else:
                    pred = int(model.predict(input_df_processed)[0])
                    result_text = 'Prematuro' if pred == 1 else 'No prematuro'
                    st.success(f"Resultado: {result_text}")
            except Exception as e:
                st.error("Error al predecir. Verifique los valores.")
                st.code(str(e))

if __name__ == "__main__":
    cargar_prediccion()