import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Sistema de Alerta Temprana (SAT)", layout="wide")

# --- 1. LISTA MAESTRA DE COLUMNAS (Orden exacto del entrenamiento) ---
COLUMNAS_MAESTRAS = [
    'EDAD', 'SEXO', 'ESTRATO', 'NATURALEZA DE COLEGIO', 
    'PUNTAJE INSCRIPCIÓN ICFES (No matriculado)', 'VALOR PAGADO', 
    'CALIF_ACADEMICA', 'CALIF_ECONOMICO', 'CALIF_FAMILIAR', 
    'CALIF_PSICOSOCIAL', 'DEPRESIÓN', 'ANSIEDAD', 'PUNTAJETABACO', 
    'PUNTAJEALCOHOL', 'PUNTAJECANNABIS', 'PUNTAJECOCAINA', 
    'PUNTAJEANFETAMINA', 'PUNTAJEINHALANTE', 'PUNTAJESEDANTE', 
    'PUNTAJEALUCINOGENO', 'PUNTAJEOPIACEO', 'PUNTAJEOTRADROGA', 
    'estrategia_ACOMODADOR', 'estrategia_ASIMILADOR', 
    'estrategia_CONVERGENTE', 'estrategia_DIVERGENTE'
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    try:
        # Cargamos los 3 archivos que generaste con joblib.dump
        sc = joblib.load(os.path.join(BASE_DIR, "modelos", "scaler.pkl"))
        m_mlp = joblib.load(os.path.join(BASE_DIR, "modelos", "mlp_model.pkl"))
        m_rf = joblib.load(os.path.join(BASE_DIR, "modelos", "random_forest_model.pkl"))
        return sc, m_mlp, m_rf
    except Exception as e:
        st.error(f"Error al cargar los archivos .pkl: {e}")
        st.stop()

scaler, mlp_modelo, rf_modelo = load_assets()

# --- 2. INTERFAZ DE USUARIO ---
st.title("🚀 Sistema de Predicción de Deserción Estudiantil")
st.markdown("Ingrese los datos del estudiante para evaluar el nivel de riesgo.")

def get_user_inputs():
    st.sidebar.header("📥 Formulario de Entrada")
    
    # Variables principales (Basadas en tus widgets de Colab)
    edad = st.sidebar.slider("Edad", 17, 70, 20)
    sexo = st.sidebar.radio("Sexo", options=[0, 1], format_func=lambda x: "Femenino" if x==0 else "Masculino")
    estrato = st.sidebar.selectbox("Estrato Socioeconómico", [1, 2, 3, 4, 5, 6], index=2)
    colegio = st.sidebar.radio("Naturaleza de Colegio", options=[0, 1], format_func=lambda x: "Público" if x==0 else "Privado")
    icfes = st.sidebar.number_input("Puntaje ICFES", 0, 500, 300)
    pago = st.sidebar.number_input("Valor Pagado Matrícula", 0, 2500000, 500000)
    nota = st.sidebar.slider("Calificación Académica (0-20)", 0.0, 20.0, 10.0)
    
    # Factores Psicosociales
    st.sidebar.subheader("Salud y Entorno")
    depresion = st.sidebar.slider("Nivel de Depresión", 0, 100, 25)
    ansiedad = st.sidebar.slider("Nivel de Ansiedad", 0, 100, 30)
    
    # Consumo (simplificado para la interfaz, el resto irá a 0)
    alcohol = st.sidebar.slider("Consumo Alcohol (Puntaje)", 0, 10, 1)
    
    # Mapeo al diccionario (debe contener las 26 llaves de COLUMNAS_MAESTRAS)
    input_data = {col: 0.0 for col in COLUMNAS_MAESTRAS} # Inicializa todo en 0
    
    input_data.update({
        'EDAD': edad, 'SEXO': sexo, 'ESTRATO': estrato, 
        'NATURALEZA DE COLEGIO': colegio, 
        'PUNTAJE INSCRIPCIÓN ICFES (No matriculado)': icfes,
        'VALOR PAGADO': pago, 'CALIF_ACADEMICA': nota,
        'DEPRESIÓN': depresion, 'ANSIEDAD': ansiedad,
        'PUNTAJEALCOHOL': alcohol
    })
    
    return pd.DataFrame([input_data])[COLUMNAS_MAESTRAS]

df_usuario = get_user_inputs()

# --- 3. LÓGICA DE PREDICCIÓN ---
if st.button("🔍 Realizar Diagnóstico"):
    try:
        # 1. ESCALAMIENTO (Crucial para MLP)
        X_scaled = scaler.transform(df_usuario.values)
        
        # 2. PREDICCIONES (Probabilidades)
        prob_mlp = mlp_modelo.predict_proba(X_scaled)[:, 1][0]
        prob_rf = rf_modelo.predict_proba(X_scaled)[:, 1][0]
        
        # Promedio de riesgo entre ambos modelos
        riesgo_final = (prob_mlp + prob_rf) / 2
        
        # --- Visualización de Resultados ---
        st.divider()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Nivel de Riesgo")
            if riesgo_final >= 0.7:
                st.error(f"⚠️ MUY ALTO: {riesgo_final:.2%}")
            elif riesgo_final >= 0.4:
                st.warning(f"🔸 MODERADO: {riesgo_final:.2%}")
            else:
                st.success(f"✅ BAJO: {riesgo_final:.2%}")
        
        with col2:
            st.subheader("Comparativa de Modelos")
            st.progress(prob_mlp, text=f"Riesgo MLP: {prob_mlp:.2%}")
            st.progress(prob_rf, text=f"Riesgo RandomForest: {prob_rf:.2%}")

        # Importancia de variables (Solo RF lo permite)
        st.subheader("📌 Factores Clave")
        importancias = pd.Series(rf_modelo.feature_importances_, index=COLUMNAS_MAESTRAS)
        top_5 = importancias.nlargest(5)
        st.bar_chart(top_5)

    except Exception as e:
        st.error(f"Error en la predicción: {e}")

# Verificador técnico (expandible)
with st.expander("Ver matriz técnica enviada al modelo"):
    st.write(df_usuario)