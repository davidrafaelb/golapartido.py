import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Configuración inicial
st.set_page_config(page_title="Estimador de Odds en Vivo", layout="wide")
st.title("⚽ Estimador de Odds en Tiempo Real")

# Parámetros del modelo (ajustables)
MEDIA_GOLES_LOCAL = 1.5
MEDIA_GOLES_VISITANTE = 1.2
VENTAJA_LOCAL = 0.3

class OddsEstimator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.goles_local = []
        self.goles_visitante = []
        self.minutos_goles = []
        self.tipo_goles = []
    
    def agregar_gol(self, minuto, tipo):
        """Agrega un gol al registro"""
        self.minutos_goles.append(minuto)
        self.tipo_goles.append(tipo)
        
        if tipo == 'L':
            self.goles_local.append(minuto)
        else:
            self.goles_visitante.append(minuto)
    
    def calcular_odds(self, minuto_actual):
        """Calcula las odds actuales basado en el tiempo transcurrido"""
        if minuto_actual <= 0:
            return {
                '1': 1.0, 'X': 1.0, '2': 1.0,
                'probabilidades': {'1': 0.33, 'X': 0.34, '2': 0.33}
            }
            
        tiempo_restante = 90 - minuto_actual
        factor_tiempo = tiempo_restante / 90
        
        goles_l = len(self.goles_local)
        goles_v = len(self.goles_visitante)
        
        # Tasa base ajustada por tiempo
        tasa_local = (MEDIA_GOLES_LOCAL + VENTAJA_LOCAL) * factor_tiempo
        tasa_visitante = MEDIA_GOLES_VISITANTE * factor_tiempo
        
        # Ajustar por el marcador actual
        if goles_l > goles_v:
            tasa_local *= 0.8
            tasa_visitante *= 1.2
        elif goles_v > goles_l:
            tasa_local *= 1.2
            tasa_visitante *= 0.8
        
        # Calcular probabilidades usando Poisson
        prob_victoria_local = 0
        prob_empate = 0
        prob_victoria_visitante = 0
        
        # Considerar goles ya marcados como base
        goles_base_local = goles_l
        goles_base_visitante = goles_v
        
        for i in range(0, 10):
            for j in range(0, 10):
                prob = poisson.pmf(i, tasa_local) * poisson.pmf(j, tasa_visitante)
                total_local = goles_base_local + i
                total_visitante = goles_base_visitante + j
                
                if total_local > total_visitante:
                    prob_victoria_local += prob
                elif total_local == total_visitante:
                    prob_empate += prob
                else:
                    prob_victoria_visitante += prob
        
        # Normalizar probabilidades (asegurar que sumen 1)
        total_prob = prob_victoria_local + prob_empate + prob_victoria_visitante
        if total_prob > 0:
            prob_victoria_local /= total_prob
            prob_empate /= total_prob
            prob_victoria_visitante /= total_prob
        
        # Convertir a odds decimales
        odds_local = 1 / prob_victoria_local if prob_victoria_local > 0.01 else 100
        odds_empate = 1 / prob_empate if prob_empate > 0.01 else 100
        odds_visitante = 1 / prob_victoria_visitante if prob_victoria_visitante > 0.01 else 100
        
        return {
            '1': odds_local,
            'X': odds_empate,
            '2': odds_visitante,
            'probabilidades': {
                '1': prob_victoria_local,
                'X': prob_empate,
                '2': prob_victoria_visitante
            }
        }

# Inicializar en session_state
if 'estimador' not in st.session_state:
    st.session_state.estimador = OddsEstimator()
if 'minuto_actual' not in st.session_state:
    st.session_state.minuto_actual = 0

estimador = st.session_state.estimador

# Sidebar para entrada de datos
with st.sidebar:
    st.header("📊 Configuración del Partido")
    
    # Slider para minuto actual
    minuto_actual = st.slider(
        "Minuto actual", 
        0, 90, 
        st.session_state.minuto_actual,
        key="minuto_slider"
    )
    
    # Actualizar el minuto en session_state
    if minuto_actual != st.session_state.minuto_actual:
        st.session_state.minuto_actual = minuto_actual
        st.rerun()
    
    st.subheader("⚽ Registrar Gol")
    col1, col2 = st.columns(2)
    with col1:
        minuto_gol = st.number_input("Minuto del gol", 0, 90, 0)
    with col2:
        tipo_gol = st.selectbox("Equipo", ['L', 'V'])
    
    if st.button("Agregar Gol"):
        estimador.agregar_gol(minuto_gol, tipo_gol)
        st.success(f"Gol de {tipo_gol} al minuto {minuto_gol} registrado")
        st.rerun()
    
    # Botón para resetear
    if st.button("🔄 Reiniciar Partido"):
        estimador.reset()
        st.session_state.minuto_actual = 0
        st.success("Partido reiniciado")
        st.rerun()
    
    # Mostrar goles registrados
    if estimador.minutos_goles:
        st.subheader("📋 Goles registrados")
        for i, (minuto, tipo) in enumerate(zip(estimador.minutos_goles, estimador.tipo_goles)):
            equipo = "Local" if tipo == 'L' else "Visitante"
            st.write(f"**{i+1}.** Min {minuto}' - {equipo}")
    else:
        st.info("No hay goles registrados")

# Panel principal
st.header(f"📊 Análisis al minuto {st.session_state.minuto_actual}")

if st.session_state.minuto_actual > 0:
    odds = estimador.calcular_odds(st.session_state.minuto_actual)
    probs = odds['probabilidades']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🏠 Victoria Local",
            value=f"{odds['1']:.2f}",
            delta=f"{probs['1']*100:.1f}%"
        )
    
    with col2:
        st.metric(
            label="⚖️ Empate",
            value=f"{odds['X']:.2f}",
            delta=f"{probs['X']*100:.1f}%"
        )
    
    with col3:
        st.metric(
            label="✈️ Victoria Visitante",
            value=f"{odds['2']:.2f}",
            delta=f"{probs['2']*100:.1f}%"
        )
    
    # Gráfico de probabilidades
    st.subheader("📈 Distribución de Probabilidades")
    fig, ax = plt.subplots(figsize=(10, 6))
    resultados = ['Local', 'Empate', 'Visitante']
    probabilidades = [probs['1'], probs['X'], probs['2']]
    colores = ['#1f77b4', '#ff7f0e', '#d62728']
    
    bars = ax.bar(resultados, probabilidades, color=colores, alpha=0.8)
    ax.set_ylabel('Probabilidad', fontsize=12)
    ax.set_title('Probabilidades de Resultado Final', fontsize=14)
    ax.set_ylim(0, 1)
    
    # Añadir valores en las barras
    for bar, prob in zip(bars, probabilidades):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.2%}', ha='center', va='bottom', fontsize=11)
    
    st.pyplot(fig)
    
    # Información adicional
    st.subheader("📋 Estadísticas del Partido")
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.info(f"**Minuto actual:** {st.session_state.minuto_actual}")
        st.info(f"**Tiempo restante:** {90 - st.session_state.minuto_actual} min")
    
    with info_col2:
        st.success(f"**Goles Local:** {len(estimador.goles_local)}")
        st.success(f"**Goles Visitante:** {len(estimador.goles_visitante)}")
    
    with info_col3:
        marcador = f"{len(estimador.goles_local)}-{len(estimador.goles_visitante)}"
        st.warning(f"**Marcador actual:** {marcador}")
        if len(estimador.goles_local) > len(estimador.goles_visitante):
            st.error("**Resultado actual:** Local gana")
        elif len(estimador.goles_visitante) > len(estimador.goles_local):
            st.error("**Resultado actual:** Visitante gana")
        else:
            st.error("**Resultado actual:** Empate")
    
else:
    st.info("⏰ Ajusta el minuto actual para comenzar el análisis")

# Explicación del modelo
with st.expander("📖 ¿Cómo funciona el modelo?"):
    st.write("""
    **Modelo Probabilístico de Odds en Tiempo Real**
    
    Este estimador utiliza:
    
    - **Distribución de Poisson**: Modela la probabilidad de que ocurran más goles
    - **Ajuste temporal**: Considera el tiempo restante de partido
    - **Ventaja de localía**: El equipo local tiene mayor probabilidad de gol
    - **Efecto del marcador**: Los equipos que van perdiendo presionan más
    
    **Fórmula base:**
    - Local: (1.5 + 0.3) × (minutos_restantes/90)
    - Visitante: 1.2 × (minutos_restantes/90)
    
    **Ajustes por marcador:**
    - Equipo que va ganando: -20% en tasa de gol
    - Equipo que va perdiendo: +20% en tasa de gol
    """)

# Debug info (opcional)
with st.expander("🔍 Debug Info"):
    st.write("Goles local:", estimador.goles_local)
    st.write("Goles visitante:", estimador.goles_visitante)
    st.write("Minutos goles:", estimador.minutos_goles)
    st.write("Tipos goles:", estimador.tipo_goles)
