import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import time
import random
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURACIÓN DE ESCENARIO ---
st.set_page_config(page_title="SAT - Inteligencia de Permanencia", page_icon="🎓", layout="wide")

# --- PALETA DE COLORES MODERNA ---
COLORS = {
    'primary': '#6366F1',
    'secondary': '#8B5CF6',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'info': '#3B82F6',
    'dark': '#1F2937',
    'light': '#F9FAFB',
    'gradient_1': 'linear-gradient(135deg, #667EEA 0%, #764BA2 100%)',
    'gradient_2': 'linear-gradient(135deg, #F093FB 0%, #F5576C 100%)',
    'gradient_3': 'linear-gradient(135deg, #4FACFE 0%, #00F2FE 100%)'
}

# --- DISEÑO Y ANIMACIONES (CSS SIN FONDOS BLANCOS) ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{ 
        font-family: 'Inter', sans-serif;
    }}
    
    .main {{ 
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
    }}
    
    /* Animación de carga mejorada */
    .brain-loader {{
        font-size: 100px;
        text-align: center;
        animation: float-brain 2s infinite ease-in-out;
        filter: drop-shadow(0 10px 20px rgba(99,102,241,0.3));
    }}
    
    @keyframes float-brain {{
        0% {{ transform: translateY(0px) scale(1); }}
        50% {{ transform: translateY(-20px) scale(1.05); }}
        100% {{ transform: translateY(0px) scale(1); }}
    }}
    
    /* Tarjetas sin fondo blanco */
    .diagnosis-card {{
        background: transparent;
        padding: 40px;
        transition: transform 0.3s ease;
    }}
    
    /* Sidebar mejorado */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #1F2937 0%, #111827 100%);
        color: white;
        border-right: 1px solid rgba(255,255,255,0.1);
    }}
    
    /* Botones con efecto */
    .stButton>button {{
        width: 100%;
        border-radius: 16px;
        height: 3.5em;
        background: {COLORS['gradient_1']};
        color: white;
        font-weight: 600;
        border: none;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(99,102,241,0.3);
    }}
    
    /* Métricas sin fondo */
    .metric-card {{
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(99,102,241,0.2);
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-3px);
        background: rgba(255,255,255,0.1);
        border-color: rgba(99,102,241,0.4);
    }}
    
    /* Badges de riesgo */
    .risk-badge-critical {{
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
        animation: pulse-red 2s infinite;
    }}
    
    .risk-badge-medium {{
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 600;
    }}
    
    .risk-badge-low {{
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 600;
    }}
    
    @keyframes pulse-red {{
        0% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0.7); }}
        70% {{ box-shadow: 0 0 0 10px rgba(239,68,68,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0); }}
    }}
    
    /* Tabs personalizadas */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 12px;
        padding: 8px 24px;
        background-color: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        color: #ffffff;
        font-weight: 500;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {COLORS['gradient_1']};
        color: white;
    }}
    
    /* Inputs mejorados */
    .stNumberInput > div > div > input, .stSelectbox > div > div, .stSlider > div {{
       border-radius: 12px;
    border: 1px solid rgba(0, 0, 0, 0.1);
    background: rgb(38, 39, 48);
    transition: all 0.3s ease;
    }}
    
    .stNumberInput > div > div > input:focus, .stSelectbox > div > div:focus {{
        border-color: {COLORS['primary']};
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1);
    }}
    
    /* Alertas modernas sin fondo blanco */
    .custom-success {{
        background: rgba(16, 185, 129, 0.1);
        backdrop-filter: blur(10px);
        border-left: 4px solid {COLORS['success']};
        padding: 16px;
        border-radius: 12px;
        margin: 10px 0;
    }}
    
    .custom-warning {{
        background: rgba(245, 158, 11, 0.1);
        backdrop-filter: blur(10px);
        border-left: 4px solid {COLORS['warning']};
        padding: 16px;
        border-radius: 12px;
        margin: 10px 0;
    }}
    
    .custom-danger {{
        background: rgba(239, 68, 68, 0.1);
        backdrop-filter: blur(10px);
        border-left: 4px solid {COLORS['danger']};
        padding: 16px;
        border-radius: 12px;
        margin: 10px 0;
    }}
    
    /* Encabezado sin fondo */
    .hero-section {{
        text-align: center;
        padding: 40px 20px;
        background: transparent;
        border-radius: 30px;
        margin-bottom: 40px;
    }}
    
    /* Info box sin fondo */
    .info-box {{
        background: rgba(99, 102, 241, 0.05);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        border: 1px solid rgba(99,102,241,0.2);
    }}
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DE MODELOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    try:
        sc = joblib.load(os.path.join(BASE_DIR, "modelos", "scaler.pkl"))
        mlp = joblib.load(os.path.join(BASE_DIR, "modelos", "mlp_model.pkl"))
        rf = joblib.load(os.path.join(BASE_DIR, "modelos", "random_forest_model.pkl"))
        return sc, mlp, rf
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None, None

scaler, mlp_modelo, rf_modelo = load_models()

COL_26 = [
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

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/null/artificial-intelligence.png", width=60)
    st.title("🎓 SAT Analytics")
    st.markdown("---")
    
    with st.expander("📊 Diccionario de Variables", expanded=True):
        st.markdown("**🎓 Académicas**")
        st.caption("• **Calificación Académica:** Promedio acumulado (0-20)")
        st.caption("• **ICFES:** Puntaje de admisión (0-500)")
        st.caption("• **Estrategia de Aprendizaje:** Perfil cognitivo")
        
        st.markdown("**🧠 Psicosociales**")
        st.caption("• **Depresión/Ansiedad:** Tamizaje emocional (0-100)")
        st.caption("• **Apoyo Familiar:** Percepción de estabilidad (-1 a 1)")
        st.caption("• **Bienestar Económico:** Situación financiera (-1 a 1)")
        
        st.markdown("**⚠️ Factores de Riesgo**")
        st.caption("• **Consumo de Sustancias:** Evaluación periódica")
        st.caption("• **Estrato Socioeconómico:** 1-6")
    
    st.divider()
    
    if 'batch_data' in st.session_state and st.session_state.batch_data is not None:
        st.markdown("### 📈 Panel Rápido")
        df_view = st.session_state.batch_data
        st.metric("Total Evaluados", len(df_view))
        st.metric("Riesgo Promedio", f"{df_view['Riesgo_%'].mean():.1f}%")
        criticos = len(df_view[df_view['Riesgo_%'] >= 70])
        if criticos > 0:
            st.error(f"⚠️ {criticos} casos críticos")

# --- NAVEGACIÓN ---
if 'view' not in st.session_state: 
    st.session_state.view = 'main'

# --- FUNCIÓN PARA GENERAR RECOMENDACIONES ---
def generar_recomendaciones(prob, datos):
    recomendaciones = []
    
    if prob >= 0.7:
        recomendaciones.append("🚨 **Intervención Inmediata Requerida:** Contactar al estudiante en las próximas 48 horas.")
        recomendaciones.append("📞 **Plan de Acción:** Asignar tutoría intensiva y seguimiento semanal.")
        
        if datos.get('dep', 0) > 60 or datos.get('ans', 0) > 60:
            recomendaciones.append("🧠 **Derivación a Salud Mental:** Remitir a servicios de psicología.")
        
        if datos.get('eco', 0) < -0.5:
            recomendaciones.append("💰 **Apoyo Financiero:** Evaluar elegibilidad para becas o fondos de emergencia.")
    
    elif prob >= 0.4:
        recomendaciones.append("🟡 **Seguimiento Prioritario:** Programar reunión de orientación en los próximos 15 días.")
        recomendaciones.append("📚 **Plan de Refuerzo:** Implementar programa de acompañamiento académico.")
        
        if datos.get('fam', 0) < -0.3:
            recomendaciones.append("👨‍👩‍👧 **Intervención Familiar:** Fortalecer la red de apoyo familiar.")
    
    else:
        recomendaciones.append("✅ **Mantenimiento Preventivo:** Seguimiento trimestral estándar.")
        recomendaciones.append("🎯 **Optimización:** Identificar áreas de oportunidad para excelencia académica.")
    
    return recomendaciones

# --- PANTALLA DE CARGA ---
if st.session_state.view == 'loading':
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='brain-loader'>🧠✨🔮</div>", unsafe_allow_html=True)
    
    msg_box = st.empty()
    bar = st.progress(0)
    frases = [
        "🧠 Activando redes neuronales...",
        "📊 Analizando patrones de comportamiento...",
        "🎓 Calculando probabilidades de permanencia...",
        "⚙️ Procesando variables psicosociales...",
        "📈 Generando insights predictivos..."
    ]
    
    wait_time = random.randint(4, 6)
    for i in range(100):
        time.sleep(wait_time/100)
        if i % 20 == 0:
            msg_box.markdown(f"""
                <div style='text-align: center; margin-top: 30px;'>
                    <p style='color: {COLORS["primary"]}; font-weight: 500;'>{random.choice(frases)}</p>
                </div>
            """, unsafe_allow_html=True)
        bar.progress(i + 1)
    
    if 'single_data' in st.session_state and st.session_state.single_data is not None:
        st.session_state.view = 'report'
    else:
        st.session_state.view = 'batch_report'
    st.rerun()

# --- REPORTE INDIVIDUAL ---
elif st.session_state.view == 'report':
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button("← Nuevo", on_click=lambda: st.session_state.update({"view": "main", "single_data": None}))
    
    d = st.session_state.get('single_data')
    if d is not None:
        reg = {col: 0.0 for col in COL_26}
        reg.update({
            'EDAD': d['edad'], 'SEXO': d['sexo'], 'ESTRATO': d['estrato'], 'NATURALEZA DE COLEGIO': d['colegio'],
            'PUNTAJE INSCRIPCIÓN ICFES (No matriculado)': d['icfes'], 'VALOR PAGADO': d['pago'],
            'CALIF_ACADEMICA': d['nota'], 'CALIF_ECONOMICO': d['eco'], 'CALIF_FAMILIAR': d['fam'],
            'CALIF_PSICOSOCIAL': d['psico'], 'DEPRESIÓN': d['dep'], 'ANSIEDAD': d['ans'],
            f"estrategia_{d['estilo'].upper()}": 1.0
        })

        X = scaler.transform(pd.DataFrame([reg])[COL_26].values)
        prob = (mlp_modelo.predict_proba(X)[:, 1][0] + rf_modelo.predict_proba(X)[:, 1][0]) / 2
        
        if prob >= 0.7:
            riesgo_texto = "CRÍTICO"
            riesgo_color = COLORS['danger']
            badge_class = "risk-badge-critical"
        elif prob >= 0.4:
            riesgo_texto = "MEDIO"
            riesgo_color = COLORS['warning']
            badge_class = "risk-badge-medium"
        else:
            riesgo_texto = "BAJO"
            riesgo_color = COLORS['success']
            badge_class = "risk-badge-low"
        
        st.markdown(f"""
            <div class='diagnosis-card'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;'>
                    <div>
                        <h1 style='margin: 0; color: {COLORS["dark"]};'>Informe de Permanencia</h1>
                        <p style='color: #6B7280; margin-top: 5px;'>Generado por SAT Intelligence System</p>
                    </div>
                    <div class='{badge_class}'>
                        🎯 RIESGO {riesgo_texto}
                    </div>
                </div>
        """, unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1, 1.5])
        
        with col_left:
            st.markdown(f"### Probabilidad de Deserción")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                number = {'suffix': "%", 'font': {'size': 40, 'color': riesgo_color}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': riesgo_color, 'thickness': 0.3},
                    'bgcolor': "rgba(255,255,255,0.1)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255,255,255,0.2)",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
                <div style='text-align: center; margin-top: 20px;'>
                    <p style='font-size: 14px; color: #6B7280;'>
                        El modelo ha analizado <strong>{len(COL_26)} variables</strong> para determinar este resultado
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col_right:
            st.markdown(f"### 📋 Análisis del Perfil")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Edad", f"{d['edad']} años")
                st.metric("ICFES", f"{d['icfes']}")
                st.metric("Promedio", f"{d['nota']:.1f}/20")
            with col_b:
                st.metric("Estrato", d['estrato'])
                st.metric("Perfil", d['estilo'])
                st.metric("Apoyo Familiar", f"{d['fam']:.1f}")
            
            st.markdown("---")
            st.markdown("### 🎯 Recomendaciones Prioritarias")
            recomendaciones = generar_recomendaciones(prob, d)
            for rec in recomendaciones:
                st.markdown(f"• {rec}")
        
        st.markdown("---")
        
        st.markdown("### 🔍 Factores de Riesgo Identificados")
        imp = pd.Series(rf_modelo.feature_importances_, index=COL_26).nlargest(5)
        
        fig_importance = go.Figure(data=[
            go.Bar(
                x=imp.values * 100,
                y=[f.replace('_', ' ').title() for f in imp.index],
                orientation='h',
                marker=dict(
                    color=[COLORS['primary'], COLORS['secondary'], COLORS['info'], COLORS['warning'], COLORS['danger']],
                    line=dict(color='white', width=1)
                ),
                text=[f'{v*100:.1f}%' for v in imp.values],
                textposition='outside'
            )
        ])
        fig_importance.update_layout(
            height=300,
            xaxis_title="Impacto en la Decisión (%)",
            yaxis_title="Variable",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.download_button(
            label="📥 Exportar Informe (JSON)",
            data=str({"probabilidad": prob, "recomendaciones": recomendaciones}),
            file_name=f"informe_sat_{d['edad']}_{d['estilo']}.json",
            mime="application/json"
        )
    else:
        st.session_state.view = 'main'
        st.rerun()

# --- REPORTE MASIVO ---
elif st.session_state.view == 'batch_report':
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button("← Nueva Carga", on_click=lambda: st.session_state.update({"view": "main", "batch_data": None}))
    
    df = st.session_state.get('batch_data')
    
    if df is not None:
        st.markdown(f"""
            <div style='background: {COLORS["gradient_1"]}; padding: 30px; border-radius: 20px; margin-bottom: 30px;'>
                <h1 style='color: white; margin: 0;'>📊 Reporte Ejecutivo de Población</h1>
                <p style='color: rgba(255,255,255,0.9); margin-top: 10px;'>
                    Análisis predictivo de {len(df)} estudiantes • Generado por SAT Intelligence
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        criticos = df[df['Riesgo_%'] >= 70]
        medios = df[(df['Riesgo_%'] >= 40) & (df['Riesgo_%'] < 70)]
        bajos = df[df['Riesgo_%'] < 40]
        
        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        
        with col_metric1:
            st.markdown(f"""
                <div class='metric-card'>
                    <p style='color: #6B7280; margin: 0;'>Total Evaluados</p>
                    <h2 style='color: {COLORS["primary"]}; margin: 10px 0;'>{len(df)}</h2>
                    <p style='font-size: 12px; color: #6B7280;'>sin duplicados</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col_metric2:
            st.markdown(f"""
                <div class='metric-card'>
                    <p style='color: #6B7280; margin: 0;'>🔴 Casos Críticos</p>
                    <h2 style='color: {COLORS["danger"]}; margin: 10px 0;'>{len(criticos)}</h2>
                    <p style='font-size: 12px; color: #6B7280;'>{len(criticos)/len(df)*100:.1f}% del total</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col_metric3:
            st.markdown(f"""
                <div class='metric-card'>
                    <p style='color: #6B7280; margin: 0;'>🟡 Riesgo Medio</p>
                    <h2 style='color: {COLORS["warning"]}; margin: 10px 0;'>{len(medios)}</h2>
                    <p style='font-size: 12px; color: #6B7280;'>{len(medios)/len(df)*100:.1f}% del total</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col_metric4:
            st.markdown(f"""
                <div class='metric-card'>
                    <p style='color: #6B7280; margin: 0;'>🟢 Bajo Riesgo</p>
                    <h2 style='color: {COLORS["success"]}; margin: 10px 0;'>{len(bajos)}</h2>
                    <p style='font-size: 12px; color: #6B7280;'>{len(bajos)/len(df)*100:.1f}% del total</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if len(criticos) > 0:
            st.markdown(f"""
                <div class='custom-danger'>
                    <strong>⚠️ ALERTA DE INTERVENCIÓN PRIORITARIA</strong><br>
                    Se han identificado <strong>{len(criticos)} estudiantes</strong> con riesgo crítico de deserción (>70%). 
                    Se recomienda activar el protocolo de atención inmediata y asignar un tutor de seguimiento.
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🚨 Listado de Casos Prioritarios")
        st.markdown("Estudiantes que requieren intervención inmediata (ordenados por nivel de riesgo)")
        
        if len(criticos) > 0:
            display_df = criticos[['Alerta', 'Riesgo_%'] + [c for c in COL_26[:5] if c in criticos.columns]].copy()
            display_df['Riesgo_%'] = display_df['Riesgo_%'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(
                display_df.sort_values('Riesgo_%', ascending=False),
                use_container_width=True,
                column_config={
                    "Alerta": st.column_config.TextColumn("Prioridad", width="small"),
                    "Riesgo_%": st.column_config.TextColumn("Riesgo", width="small")
                }
            )
        else:
            st.success("✅ No se encontraron casos críticos en esta población.")
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### 📈 Distribución de Riesgo")
            fig_pie = px.pie(
                df, 
                names='Alerta', 
                color='Alerta',
                color_discrete_map={
                    '🔴 CRÍTICO': COLORS['danger'],
                    '🟡 MEDIO': COLORS['warning'],
                    '🟢 BAJO': COLORS['success']
                },
                hole=0.4,
                title="Distribución por Nivel de Riesgo"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_chart2:
            st.markdown("### 📊 Histograma de Probabilidades")
            fig_hist = px.histogram(
                df,
                x='Riesgo_%',
                nbins=20,
                title="Distribución de Probabilidades de Deserción",
                color_discrete_sequence=[COLORS['primary']]
            )
            fig_hist.add_vline(x=40, line_dash="dash", line_color=COLORS['warning'], 
                               annotation_text="Riesgo Medio", annotation_position="top")
            fig_hist.add_vline(x=70, line_dash="dash", line_color=COLORS['danger'],
                               annotation_text="Riesgo Crítico", annotation_position="top")
            fig_hist.update_layout(
                xaxis_title="Probabilidad de Deserción (%)",
                yaxis_title="Número de Estudiantes",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("### 📊 Resumen por Rangos")
        summary_data = {
            'Nivel': ['Crítico (≥70%)', 'Medio (40-69%)', 'Bajo (<40%)'],
            'Cantidad': [len(criticos), len(medios), len(bajos)],
            'Porcentaje': [f"{len(criticos)/len(df)*100:.1f}%", f"{len(medios)/len(df)*100:.1f}%", f"{len(bajos)/len(df)*100:.1f}%"],
            'Riesgo Promedio': [
                f"{criticos['Riesgo_%'].mean():.1f}%" if len(criticos) > 0 else "N/A",
                f"{medios['Riesgo_%'].mean():.1f}%" if len(medios) > 0 else "N/A",
                f"{bajos['Riesgo_%'].mean():.1f}%" if len(bajos) > 0 else "N/A"
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            st.download_button(
                "📥 Reporte Completo (CSV)",
                df.to_csv(index=False).encode('utf-8'),
                "reporte_sat_completo.csv",
                mime="text/csv"
            )
        with col_btn2:
            if len(criticos) > 0:
                st.download_button(
                    "🚨 Casos Críticos (CSV)",
                    criticos.to_csv(index=False).encode('utf-8'),
                    "casos_criticos_sat.csv",
                    mime="text/csv"
                )
        with col_btn3:
            resumen = f"""
            INFORME EJECUTIVO SAT
            ======================
            Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            Total de estudiantes evaluados: {len(df)}
            
            Distribución de riesgo:
            - Crítico (>70%): {len(criticos)} ({len(criticos)/len(df)*100:.1f}%)
            - Medio (40-70%): {len(medios)} ({len(medios)/len(df)*100:.1f}%)
            - Bajo (<40%): {len(bajos)} ({len(bajos)/len(df)*100:.1f}%)
            
            Riesgo promedio general: {df['Riesgo_%'].mean():.1f}%
            
            Recomendaciones:
            - Atención prioritaria a casos críticos
            - Programas preventivos para riesgo medio
            - Seguimiento regular a bajo riesgo
            """
            st.download_button(
                "📋 Resumen Ejecutivo (TXT)",
                resumen,
                "resumen_ejecutivo_sat.txt",
                mime="text/plain"
            )
    else:
        st.error("Error al procesar el lote. Intente de nuevo.")

# --- PANTALLA PRINCIPAL ---
else:  # st.session_state.view == 'main'
    st.markdown(f"""
        <div class='hero-section'>
            <h1 style='font-size: 3em; margin: 0; background: {COLORS["gradient_1"]}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>
                🛡️ SAT Intelligence System
            </h1>
            <p style='font-size: 1.2em; color: #6B7280; margin-top: 15px;'>
                Predicción Inteligente de Permanencia Estudiantil
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    t1, t2 = st.tabs(["👤 Evaluación Individual", "📂 Análisis Masivo"])
    
    with t1:
        st.markdown("""
            <div class='info-box'>
                <p style='margin: 0; color: #4B5563;'>📝 Complete la información del estudiante para obtener un análisis detallado
                del riesgo de deserción y recomendaciones personalizadas.</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("form_ind"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📊 Datos Sociodemográficos")
                edad = st.number_input("Edad", 16, 85, 20)
                sexo = st.radio("Sexo", [0, 1], format_func=lambda x: "👩 Femenino" if x==0 else "👨 Masculino", horizontal=True)
                estrato = st.selectbox("🏠 Estrato Socioeconómico", [1,2,3,4,5,6], index=1)
                pago = st.number_input("💰 Valor Matrícula", 0, 8000000, 750000)
                colegio = st.radio("🏫 Tipo de Institución", [0,1], 
                                  format_func=lambda x: "Pública" if x==0 else "Privada", horizontal=True)
            
            with col2:
                st.markdown("#### 🎓 Rendimiento Académico")
                icfes = st.number_input("📝 Puntaje ICFES", 0, 500, 280)
                nota = st.slider("⭐ Promedio Académico", 0.0, 20.0, 11.0, format="%.1f")
                estilo = st.selectbox("🧠 Estilo de Aprendizaje", 
                                      ["Acomodador", "Asimilador", "Convergente", "Divergente"])
                fam = st.slider("👨‍👩‍👧 Apoyo Familiar", -1.0, 1.0, 0.0)
                eco = st.slider("💰 Situación Económica", -1.0, 1.0, 0.0)
            
            with col3:
                st.markdown("#### 🌟 Bienestar Estudiantil")
                ans = st.slider("😰 Nivel de Ansiedad", 0, 100, 20)
                dep = st.slider("😔 Nivel de Depresión", 0, 100, 20)
                psico = st.slider("🤝 Integración Social", -1.0, 1.0, 0.0)
                
                st.markdown("---")
                st.markdown("#### 🚭 Consumo de Sustancias")
                sust = st.multiselect("Sustancias consumidas", 
                                     ["Alcohol", "Tabaco", "Marihuana", "Otras"])
                frec = st.slider("Frecuencia de consumo (0-10)", 0, 10, 0)
            
            st.markdown("---")
            if st.form_submit_button("🛡️ ANALIZAR PERFIL", use_container_width=True):
                st.session_state.single_data = locals()
                st.session_state.view = 'loading'
                st.rerun()
    
    with t2:
        st.markdown("""
            <div class='info-box'>
                <p style='margin: 0; color: #4B5563;'>📂 Cargue un archivo con múltiples estudiantes para obtener un análisis
                poblacional completo, incluyendo distribución de riesgos y casos prioritarios.</p>
            </div>
        """, unsafe_allow_html=True)
        
        archivo = st.file_uploader(
            "📁 Subir archivo de datos",
            type=["xlsx", "csv"],
            help="Formatos soportados: Excel (.xlsx) o CSV (.csv). El archivo debe contener las 26 variables requeridas."
        )
        
        if archivo:
            st.info("🔍 Procesando archivo...")
            df_raw = pd.read_excel(archivo) if archivo.name.endswith('xlsx') else pd.read_csv(archivo)
            
            with st.expander("📋 Vista previa del archivo"):
                st.dataframe(df_raw.head(), use_container_width=True)
                st.caption(f"Total de registros encontrados: {len(df_raw)}")
            
            original_len = len(df_raw)
            df_raw = df_raw.drop_duplicates()
            if len(df_raw) < original_len:
                st.warning(f"⚠️ Se eliminaron {original_len - len(df_raw)} registros duplicados.")
            
            if all(c in df_raw.columns for c in COL_26):
                if st.button("🚀 INICIAR ANÁLISIS MASIVO", use_container_width=True):
                    with st.spinner("🧠 Procesando..."):
                        time.sleep(1)
                        X_b = scaler.transform(df_raw[COL_26].values)
                        scores = (mlp_modelo.predict_proba(X_b)[:, 1] + rf_modelo.predict_proba(X_b)[:, 1]) / 2
                        
                        df_raw['Riesgo_%'] = (scores * 100).round(1)
                        df_raw['Alerta'] = np.where(scores >= 0.7, "🔴 CRÍTICO", 
                                                   np.where(scores >= 0.4, "🟡 MEDIO", "🟢 BAJO"))
                        
                        st.session_state.batch_data = df_raw
                        st.session_state.view = 'loading'
                        st.rerun()
            else:
                columnas_faltantes = [c for c in COL_26 if c not in df_raw.columns]
                st.error(f"❌ Faltan columnas: {', '.join(columnas_faltantes[:5])}")