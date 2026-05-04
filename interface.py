import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import time

# --- CONFIGURACIÓN DE PÁGINA PROFESIONAL ---
st.set_page_config(
    page_title="SAT - Sistema de Alerta Temprana",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS UX AVANZADOS (CSS) ---
st.markdown("""
    <style>
    /* Fondo y tipografía general */
    .main { background-color: #f0f2f6; }
    
    /* Estilo de las tarjetas de métricas */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #2E86C1;
    }

    /* Botones principales */
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        height: 3em;
        background-color: #2E86C1;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #21618C;
        border: none;
    }

    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #ffffff; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DE INTELIGENCIA (MODELOS) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_ai_engine():
    try:
        scaler = joblib.load(os.path.join(BASE_DIR, "modelos", "scaler.pkl"))
        mlp = joblib.load(os.path.join(BASE_DIR, "modelos", "mlp_model.pkl"))
        rf = joblib.load(os.path.join(BASE_DIR, "modelos", "random_forest_model.pkl"))
        return scaler, mlp, rf
    except:
        st.error("⚠️ Error crítico: No se encuentran los motores de IA en la carpeta /modelos.")
        return None, None, None

scaler, mlp_modelo, rf_modelo = load_ai_engine()

# --- ESTRUCTURA DE DATOS (26 VARIABLES) ---
COLUMNAS = [
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

# --- INTERFAZ LATERAL (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=100)
    st.title("Panel de Control")
    st.markdown("---")
    st.info("**Objetivo:** Prevenir la deserción mediante la identificación temprana de factores de riesgo.")
    modo = st.radio("Herramienta:", ["🔍 Diagnóstico Individual", "📂 Procesamiento Masivo", "📊 Diccionario Técnico"])

# --- CONTENIDO PRINCIPAL ---
if modo == "🔍 Diagnóstico Individual":
    st.header("Análisis de Riesgo Estudiantil")
    st.markdown("Complete el perfil integral para obtener la probabilidad de deserción.")

    with st.form("form_psico"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.subheader("👤 Sociodemográfico")
            edad = st.number_input("Edad Actual", 15, 80, 19)
            sexo = st.selectbox("Sexo Biológico", [0, 1], format_func=lambda x: "Femenino" if x==0 else "Masculino")
            estrato = st.select_slider("Estrato", options=[1, 2, 3, 4, 5, 6], value=2)
            colegio = st.radio("Tipo de Colegio", [0, 1], format_func=lambda x: "Público" if x==0 else "Privado")
            pago = st.number_input("Valor Matrícula ($)", 0, 5000000, 500000)

        with c2:
            st.subheader("📚 Académico y Entorno")
            icfes = st.number_input("Puntaje ICFES", 0, 500, 250)
            nota = st.slider("Nota Académica (0-20)", 0.0, 20.0, 11.5)
            estilo = st.selectbox("Estilo de Aprendizaje", ["Acomodador", "Asimilador", "Convergente", "Divergente"])
            familiar = st.slider("Estabilidad Familiar (-1 a 1)", -1.0, 1.0, 0.0)
            economico = st.slider("Estabilidad Económica (-1 a 1)", -1.0, 1.0, 0.0)

        with c3:
            st.subheader("🧠 Salud Mental")
            ansiedad = st.slider("Nivel de Ansiedad", 0, 100, 20)
            depresion = st.slider("Nivel de Depresión", 0, 100, 20)
            psico = st.slider("Factor Psicosocial (-1 a 1)", -1.0, 1.0, 0.0)
            sustancias = st.multiselect("Reporte de Consumo", ["Alcohol", "Tabaco", "Cannabis", "Otras"])
            intensidad = st.slider("Frecuencia de Consumo", 0, 10, 0)

        submit = st.form_submit_button("ANALIZAR PERFIL")

    if submit:
        # Construcción del registro de 26 variables
        data = {col: 0.0 for col in COLUMNAS}
        data.update({
            'EDAD': edad, 'SEXO': sexo, 'ESTRATO': estrato, 'NATURALEZA DE COLEGIO': colegio,
            'PUNTAJE INSCRIPCIÓN ICFES (No matriculado)': icfes, 'VALOR PAGADO': pago,
            'CALIF_ACADEMICA': nota, 'CALIF_ECONOMICO': economico, 'CALIF_FAMILIAR': familiar,
            'CALIF_PSICOSOCIAL': psico, 'DEPRESIÓN': depresion, 'ANSIEDAD': ansiedad,
            f'estrategia_{estilo.upper()}': 1.0
        })
        
        # Mapeo de sustancias
        for s in sustancias:
            key = f"PUNTAJE{s.upper()}"
            if key in data: data[key] = float(intensidad)

        # Predicción
        with st.spinner("La IA está procesando el perfil..."):
            df = pd.DataFrame([data])[COLUMNAS]
            X = scaler.transform(df.values)
            p_mlp = mlp_modelo.predict_proba(X)[:, 1][0]
            p_rf = rf_modelo.predict_proba(X)[:, 1][0]
            riesgo = (p_mlp + p_rf) / 2
            time.sleep(1)

        # Panel de Resultados
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if riesgo >= 0.7:
                st.error(f"## RIESGO CRÍTICO\n**Probabilidad: {riesgo:.1%}**")
            elif riesgo >= 0.4:
                st.warning(f"## RIESGO MODERADO\n**Probabilidad: {riesgo:.1%}**")
            else:
                st.success(f"## RIESGO BAJO\n**Probabilidad: {riesgo:.1%}**")
        
        with res_col2:
            st.markdown("### 💡 Diagnóstico Narrativo")
            importancias = pd.Series(rf_modelo.feature_importances_, index=COLUMNAS)
            top_3 = importancias.nlargest(3).index.tolist()
            
            st.write(f"El análisis sugiere que el riesgo está impulsado principalmente por: **{top_3[0]}**, **{top_3[1]}** y **{top_3[2]}**.")
            st.info(f"**Protocolo sugerido:** Remitir a entrevista de Bienestar para validar el factor de {top_3[0].lower().replace('_', ' ')}.")

elif modo == "📂 Procesamiento Masivo":
    st.header("Carga Masiva de Estudiantes")
    st.write("Suba un archivo con los datos de múltiples estudiantes (26 columnas) para generar un reporte de alertas.")
    
    file = st.file_uploader("Arrastre su archivo Excel o CSV aquí", type=["xlsx", "csv"])
    
    if file:
        df_batch = pd.read_excel(file) if file.name.endswith('xlsx') else pd.read_csv(file)
        
        if all(c in df_batch.columns for c in COLUMNAS):
            with st.spinner("Procesando base de datos..."):
                X_b = scaler.transform(df_batch[COLUMNAS].values)
                probs = (mlp_modelo.predict_proba(X_b)[:, 1] + rf_modelo.predict_proba(X_b)[:, 1]) / 2
                df_batch['Riesgo_%'] = (probs * 100).round(2)
                df_batch['Alerta'] = np.where(probs >= 0.7, "🔴 CRÍTICO", np.where(probs >= 0.4, "🟡 MODERADO", "🟢 BAJO"))
            
            st.success(f"Se procesaron {len(df_batch)} registros.")
            st.dataframe(df_batch[['Alerta', 'Riesgo_%'] + COLUMNAS[:5]].sort_values('Riesgo_%', ascending=False))
            
            csv = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar Reporte Final", csv, "reporte_sat_completo.csv", "text/csv")
        else:
            st.error("El archivo no contiene las 26 columnas requeridas. Por favor, revise el Diccionario Técnico.")

elif modo == "📊 Diccionario Técnico":
    st.header("Guía de Variables para el Usuario")
    st.markdown("""
    | Categoría | Variable | Descripción |
    | :--- | :--- | :--- |
    | **Académico** | `CALIF_ACADEMICA` | Promedio del estudiante (Escala 0-20). |
    | **Psicológico** | `DEPRESIÓN` / `ANSIEDAD` | Nivel de sintomatología detectada (0-100). |
    | **Económico** | `VALOR PAGADO` | Costo de la matrícula financiada. |
    | **Estilo** | `Estrategia_...` | Perfil cognitivo dominante del estudiante. |
    """)
    st.info("💡 Consejo: Los valores en escala de -1 a 1 representan la autopercepción del estudiante (Siendo -1 muy malo y 1 excelente).")