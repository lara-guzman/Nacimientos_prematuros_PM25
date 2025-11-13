import streamlit as st
import pandas as pd
from paginas.contenido import cargar_pm25, cargar_nacimientos
from paginas.portada import cargar_portada
from paginas.prediccion import cargar_prediccion
from pathlib import Path
from joblib import load
import io

# Menu lateral para seleccionar página
pagina = st.sidebar.selectbox("Selecciona la página", ["Home", "Datos PM2.5", "Registros de Nacimientos", "Predicción Prematuridad"])

# Navegación simple
if pagina == "Home":
    cargar_portada()
elif pagina == "Datos PM2.5":
    cargar_pm25()
elif pagina == "Registros de Nacimientos":
    cargar_nacimientos()
elif pagina == "Predicción Prematuridad":
    cargar_prediccion()
