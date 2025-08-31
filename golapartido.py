import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson, expon
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    def calcular_tasas_gol(self, minuto_actual):
        """Calcula las tasas de gol actualizadas"""
        tiempo_restante = max(90 - minuto_actual, 1)
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
        
        return tasa_local, tasa_visitante
    
    def calcular_odds(self, minuto_actual):
        """Calcula las odds actuales basado en el tiempo transcurrido"""
        if minuto_actual <= 0 or minuto_actual >= 90:
            return {
                '1': 1.0, 'X': 1.0, '2': 1.0,
                'probabilidades': {'1': 0.33, 'X': 0.34, '2': 0.33}
            }
            
        tasa_local, tasa_visitante = self.calcular_tasas_gol(minuto_actual)
        
        goles_l = len(self.goles_local)
        goles_v = len(self.goles_visitante)
        
        # Calcular probabilidades usando Poisson
        prob_victoria_local = 0
        prob_empate = 0
        prob_victoria_visitante = 0
        
        for i in range(0, 10):
            for j in range(0, 10):
                prob = poisson.pmf(i, tasa_local) * poisson.pmf(j, tasa_visitante)
                total_local = goles_l + i
                total_visitante = goles_v + j
                
                if total_local > total_visitante:
                    prob_victoria_local += prob
                elif total_local == total_visitante:
                    prob_empate += prob
                else:
                    prob_victoria_visitante += prob
        
        # Normalizar probabilidades
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
            },
            'tasas': {
                'local': tasa_local,
                'visitante': tasa_visitante
            }
        }
    
    def estimar_proximos_goles(self, minuto_actual):
        """Estima los minutos probables de los próximos goles"""
        if minuto_actual >= 90:
            return {'local': [], 'visitante': []}
            
        tasa_local, tasa_visitante = self.calcular_tasas_gol(minuto_actual)
        tiempo_restante = 90 - minuto_actual
        
        # Calcular intensidad de goles (goles por minuto)
        intensidad_local = tasa_local / tiempo_restante
        intensidad_visitante = tasa_visitante / tiempo_restante
        
        # Estimación de próximos goles usando distribución exponencial
        goles_estimados = {'local': [], 'visitante': []}
        
        # Para local
        tiempo_acumulado = 0
        while tiempo_acumulado < tiempo_restante and len(goles_estimados['local']) < 3:
            # Tiempo hasta próximo gol (distribución exponencial)
            tiempo_hasta_gol = np.random.exponential(1/intensidad_local) if intensidad_local > 0 else 1000
            minuto_gol = minuto_actual + tiempo_acumulado + tiempo_hasta_gol
            
            if minuto_gol <= 90:
                goles_estimados['local'].append(minuto_gol)
                tiempo_acumulado += tiempo_hasta_gol
            else:
                break
        
        # Para visitante
        tiempo_acumulado = 0
        while tiempo_acumulado < tiempo_restante and len(goles_estimados['visitante']) < 3:
            tiempo_hasta_gol = np.random.exponential(1/intensidad_visitante) if intensidad_visitante > 0 else 1000
            minuto_gol = minuto_actual + tiempo_acumulado + tiempo_hasta_gol
            
            if minuto_gol <= 90:
                goles_estimados['visitante'].append(minuto_gol)
                tiempo_acumulado += tiempo_hasta_gol
            else:
                break
        
        # Hacer múltiples simulaciones para mejor estimación
        simulaciones = 1000
        promedios_local = []
        promedios_visitante = []
        
        for _ in range(simulaciones):
            goles_sim = self._simular_goles(minuto_actual, intensidad_local, intensidad_visitante)
            if goles_sim['local']:
                promedios_local.extend(goles_sim['local'])
            if goles_sim['visitante']:
                promedios_visitante.extend(goles_sim['visitante'])
        
        # Calcular percentiles para los goles más probables
        estimacion_final = {'local': [], 'visitante': []}
        
        if promedios_local:
            estimacion_final['local'] = [
                np.percentile(promedios_local, 25),  # Primer gol más probable
                np.percentile(promedios_local, 50),  # Segundo gol
                np.percentile(promedios_local, 75)   # Tercer gol
            ]
        
        if promedios_visitante:
            estimacion_final['visitante'] = [
                np.percentile(promedios_visitante, 25),
                np.percentile(promedios_visitante, 50),
                np.percentile(promedios_visitante, 75)
            ]
        
        return estimacion_final
    
    def _simular_goles(self, minuto_actual, intensidad_local, intensidad_visitante):
        """Simula goles para una iteración"""
        tiempo_restante = 90 - minuto_actual
        goles_simulados = {'local': [], 'visitante': []}
        
        # Simular goles local
        tiempo = 0
        while tiempo < tiempo_restante:
            tiempo_hasta_gol = np.random.exponential(1/intensidad_local) if intensidad_local > 0 else 1000
            if tiempo + tiempo_hasta_gol < tiempo_restante:
                goles_simulados['local'].append(minuto_actual + tiempo + tiempo_hasta_gol)
                tiempo += tiempo_hasta_gol
            else:
                break
        
        # Simular goles visitante
        tiempo = 0
        while tiempo < tiempo_restante:
            tiempo_hasta_gol = np.random.exponential(1/intensidad_visitante) if intensidad_visitante > 0 else 1000
            if tiempo + tiempo_hasta_gol < tiempo_restante:
                goles_simulados['visitante'].append(minuto_actual + tiempo + tiempo_hasta_gol)
                tiempo += tiempo_hasta_gol
            else:
                break
        
        return goles_simulados

# Inicializar en session_state
if 'estimador' not in st.session_state:
    st.session_state.estimador = OddsEstimator()
if 'minuto_actual' not in st.session_state:
    st.session_state.minuto_actual = 0

estimador = st.session_state.estimador

# Sidebar para entrada de datos
with st.sidebar:
    st.header("📊 Configuración del Partido")
    
    minuto_actual = st.slider(
        "Minuto actual", 
        0, 90, 
        st.session_state.minuto_actual,
        key="minuto_slider"
    )
    
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
    
    if st.button("🔄 Reiniciar Partido"):
        estimador.reset()
        st.session_state.minuto_actual = 0
        st.success("Partido reiniciado")
        st.rerun()
    
    if estimador.minutos_goles:
        st.subheader("📋 Goles registrados")
        for i, (minuto, tipo) in enumerate(zip(estimador.minutos_goles, estimador.tipo_goles)):
            equipo = "Local" if tipo == 'L' else "Visitante"
            st.write(f"**{i+1}.** Min {minuto}' - {equipo}")

# Panel principal
st.header(f"📊 Análisis al minuto {st.session_state.minuto_actual}")

if st.session_state.minuto_actual > 0 and st.session_state.minuto_actual < 90:
    odds = estimador.calcular_odds(st.session_state.minuto_actual)
    probs = odds['probabilidades']
    
    # Estimación de próximos goles
    proximos_goles = estimador.estimar_proximos_goles(st.session_state.minuto_actual)
    
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
    
    # Sección de estimación de próximos goles
    st.subheader("⏰ Estimación de Próximos Goles")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("### 🏠 Local")
        if proximos_goles['local']:
            for i, minuto in enumerate(proximos_goles['local']):
                prob_gol = (1 - (i * 0.3)) * 0.7  # Probabilidad decreciente
                st.info(f"**Gol {i+1}:** Minuto {minuto:.1f}' ({(prob_gol*100):.0f}% prob)")
        else:
            st.warning("Baja probabilidad de más goles")
    
    with col_g2:
        st.markdown("### ✈️ Visitante")
        if proximos_goles['visitante']:
            for i, minuto in enumerate(proximos_goles['visitante']):
                prob_gol = (1 - (i * 0.3)) * 0.7
                st.info(f"**Gol {i+1}:** Minuto {minuto:.1f}' ({(prob_gol*100):.0f}% prob)")
        else:
            st.warning("Baja probabilidad de más goles")
    
    # Gráfico de línea de tiempo de goles
    st.subheader("📈 Línea de Tiempo de Goles")
    
    # Crear datos para el gráfico
    minutos = list(range(0, 91, 5))
    prob_goles_local = []
    prob_goles_visitante = []
    
    tasa_local, tasa_visitante = estimador.calcular_tasas_gol(st.session_state.minuto_actual)
    
    for minuto in minutos:
        if minuto > st.session_state.minuto_actual:
            tiempo_restante = 90 - minuto
            prob_local = 1 - poisson.cdf(0, tasa_local * (minuto - st.session_state.minuto_actual) / (90 - st.session_state.minuto_actual))
            prob_visitante = 1 - poisson.cdf(0, tasa_visitante * (minuto - st.session_state.minuto_actual) / (90 - st.session_state.minuto_actual))
            prob_goles_local.append(prob_local)
            prob_goles_visitante.append(prob_visitante)
        else:
            prob_goles_local.append(0)
            prob_goles_visitante.append(0)
    
    # Crear gráfico con Plotly
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    fig.add_trace(go.Scatter(
        x=minutos, y=prob_goles_local,
        name='Probabilidad Gol Local',
        line=dict(color='blue', width=3),
        fill='tozeroy'
    ))
    
    fig.add_trace(go.Scatter(
        x=minutos, y=prob_goles_visitante,
        name='Probabilidad Gol Visitante',
        line=dict(color='red', width=3),
        fill='tozeroy'
    ))
    
    # Marcar goles existentes
    for minuto in estimador.goles_local:
        fig.add_vline(x=minuto, line_dash="dash", line_color="blue", opacity=0.7)
    
    for minuto in estimador.goles_visitante:
        fig.add_vline(x=minuto, line_dash="dash", line_color="red", opacity=0.7)
    
    fig.update_layout(
        title='Probabilidad de Gol por Minuto',
        xaxis_title='Minuto',
        yaxis_title='Probabilidad',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Información adicional
    st.subheader("📋 Estadísticas del Partido")
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.info(f"**Minuto actual:** {st.session_state.minuto_actual}")
        st.info(f"**Tiempo restante:** {90 - st.session_state.minuto_actual} min")
    
    with info_col2:
        st.success(f"**Goles Local:** {len(estimador.goles_local)}")
        st.success(f"**Goles Visitante:** {len(estimador.goles_visitante)}")
        st.success(f"**Tasa gol/local:** {odds['tasas']['local']:.2f}")
        st.success(f"**Tasa gol/visitante:** {odds['tasas']['visitante']:.2f}")
    
    with info_col3:
        marcador = f"{len(estimador.goles_local)}-{len(estimador.goles_visitante)}"
        st.warning(f"**Marcador actual:** {marcador}")
        
elif st.session_state.minuto_actual >= 90:
    st.success("🎉 Partido Finalizado")
    st.write(f"**Resultado Final:** {len(estimador.goles_local)}-{len(estimador.goles_visitante)}")
else:
    st.info("⏰ Ajusta el minuto actual para comenzar el análisis")

# Explicación del modelo
with st.expander("📖 ¿Cómo funciona la estimación de goles?"):
    st.write("""
    **Estimación de Minutos de Gol:**
    
    - **Distribución Exponencial**: Modela el tiempo entre eventos (goles)
    - **Intensidad de Gol**: Goles esperados por minuto basado en:
      - Tasa histórica del equipo
      - Marcador actual
      - Tiempo restante
      - Ventaja de localía
    
    - **Simulación Monte Carlo**: 1000 simulaciones para estimar los minutos más probables
    - **Percentiles**: Se muestran los minutos del percentil 25, 50 y 75
    
    **Interpretación:**
    - Gol 1: Más probable (25% de las simulaciones)
    - Gol 2: Mediana de probabilidad (50%)
    - Gol 3: Menos probable pero posible (75%)
    """)
