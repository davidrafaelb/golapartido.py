import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
from datetime import datetime

# Configuración inicial
st.set_page_config(page_title="Estimador de Odds en Vivo", layout="wide")
st.title("⚽ Estimador de Odds en Tiempo Real")

# Parámetros del modelo (ajustables)
MEDIA_GOLES_LOCAL = 1.5  # Goles esperados por local
MEDIA_GOLES_VISITANTE = 1.2  # Goles esperados por visitante
VENTAJA_LOCAL = 0.3  # Ventaja adicional por jugar en casa

class OddsEstimator:
    def __init__(self):
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
        tiempo_restante = 90 - minuto_actual
        tiempo_transcurrido = minuto_actual
        
        # Ajustar tasas de gol basado en el tiempo jugado
        factor_tiempo = tiempo_restante / 90
        
        # Ajustar por goles ya marcados (equipo que va ganando puede bajar ritmo)
        goles_l = len(self.goles_local)
        goles_v = len(self.goles_visitante)
        
        # Tasa ajustada de goles
        tasa_local = (MEDIA_GOLES_LOCAL + VENTAJA_LOCAL) * factor_tiempo
        tasa_visitante = MEDIA_GOLES_VISITANTE * factor_tiempo
        
        # Ajustar por el marcador actual
        if goles_l > goles_v:
            tasa_local *= 0.8  # Local gana, puede bajar ritmo
            tasa_visitante *= 1.2  # Visitante presiona
        elif goles_v > goles_l:
            tasa_local *= 1.2
            tasa_visitante *= 0.8
        
        # Calcular probabilidades usando Poisson
        prob_victoria_local = 0
        prob_empate = 0
        prob_victoria_visitante = 0
        
        for i in range(0, 10):  # Máximo 10 goles por equipo
            for j in range(0, 10):
                prob = poisson.pmf(i, tasa_local) * poisson.pmf(j, tasa_visitante)
                if i + goles_l > j + goles_v:
                    prob_victoria_local += prob
                elif i + goles_l == j + goles_v:
                    prob_empate += prob
                else:
                    prob_victoria_visitante += prob
        
        # Convertir probabilidades a odds decimales
        odds_local = 1 / prob_victoria_local if prob_victoria_local > 0 else 100
        odds_empate = 1 / prob_empate if prob_empate > 0 else 100
        odds_visitante = 1 / prob_victoria_visitante if prob_victoria_visitante > 0 else 100
        
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

# Interfaz de Streamlit
estimador = OddsEstimator()

# Sidebar para entrada de datos
with st.sidebar:
    st.header("📊 Configuración del Partido")
    
    minuto_actual = st.slider("Minuto actual", 0, 90, 0)
    
    st.subheader("⚽ Registrar Gol")
    col1, col2 = st.columns(2)
    with col1:
        minuto_gol = st.number_input("Minuto del gol", 0, 90, 0)
    with col2:
        tipo_gol = st.selectbox("Equipo", ['L', 'V'])
    
    if st.button("Agregar Gol"):
        estimador.agregar_gol(minuto_gol, tipo_gol)
        st.success(f"Gol de {tipo_gol} al minuto {minuto_gol} registrado")
    
    # Mostrar goles registrados
    if estimador.minutos_goles:
        st.subheader("Goles registrados")
        for i, (minuto, tipo) in enumerate(zip(estimador.minutos_goles, estimador.tipo_goles)):
            st.write(f"{i+1}. Min {minuto}' - {'Local' if tipo == 'L' else 'Visitante'}")

# Panel principal
col1, col2, col3 = st.columns(3)

if minuto_actual > 0:
    odds = estimador.calcular_odds(minuto_actual)
    probs = odds['probabilidades']
    
    with col1:
        st.metric(
            label="Victoria Local",
            value=f"{odds['1']:.2f}",
            delta=f"{probs['1']*100:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Empate",
            value=f"{odds['X']:.2f}",
            delta=f"{probs['X']*100:.1f}%"
        )
    
    with col3:
        st.metric(
            label="Victoria Visitante",
            value=f"{odds['2']:.2f}",
            delta=f"{probs['2']*100:.1f}%"
        )
    
    # Gráfico de probabilidades
    fig, ax = plt.subplots()
    resultados = ['Local', 'Empate', 'Visitante']
    probabilidades = [probs['1'], probs['X'], probs['2']]
    
    bars = ax.bar(resultados, probabilidades, color=['blue', 'gray', 'red'])
    ax.set_ylabel('Probabilidad')
    ax.set_title('Probabilidades de Resultado')
    ax.set_ylim(0, 1)
    
    # Añadir valores en las barras
    for bar, prob in zip(bars, probabilidades):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.2%}', ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Información adicional
    st.subheader("📈 Información del Partido")
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.write(f"**Minuto actual:** {minuto_actual}")
        st.write(f"**Goles Local:** {len(estimador.goles_local)}")
        st.write(f"**Goles Visitante:** {len(estimador.goles_visitante)}")
    
    with info_col2:
        st.write(f"**Tiempo restante:** {90 - minuto_actual} minutos")
        st.write(f"**Marcador actual:** {len(estimador.goles_local)}-{len(estimador.goles_visitante)}")
    
else:
    st.info("Ajusta el minuto actual para comenzar el análisis")

# Explicación del modelo
with st.expander("📖 Explicación del Modelo"):
    st.write("""
    Este estimador utiliza:
    - **Distribución de Poisson** para modelar la probabilidad de goles
    - **Ajuste por tiempo** que considera el tiempo restante
    - **Factor de localía** que da ventaja al equipo local
    - **Ajuste por marcador** que modifica las tasas según el resultado actual
    
    Los parámetros pueden ajustarse en el código para mayor precisión.
    """)
