import streamlit as st
from pathlib import Path
import pandas as pd

from paginas.contenido import cargar_pm25, cargar_nacimientos

def _base_path():
    return Path(__file__).resolve().parent.parent

    base = _base_path()
    results = {}
    try:
        df_nac = pd.read_csv(base / "datasets" / "consolidado.csv", encoding="utf-8", nrows=10)
        results["nacimientos_ok"] = True
        # intentar obtener número total de filas sin leer todo el archivo (fallback simple)
        try:
            results["nacimientos_count"] = sum(1 for _ in open(base / "datasets" / "consolidado.csv", "r", encoding="utf-8")) - 1
        except Exception:
            results["nacimientos_count"] = None
    except Exception:
        results["nacimientos_ok"] = False
        results["nacimientos_count"] = None

    try:
        df_pm = pd.read_csv(base / "datasets" / "pm25_mensual_municipio_202511071445.csv", nrows=10)
        results["pm25_ok"] = True
        try:
            results["pm25_count"] = sum(1 for _ in open(base / "datasets" / "pm25_mensual_municipio_202511071445.csv", "r")) - 1
        except Exception:
            results["pm25_count"] = None
    except Exception:
        results["pm25_ok"] = False
        results["pm25_count"] = None

    return results

# Funciones para cada página
def cargar_portada():
    st.title("Proyecto: PM2.5 y Nacimientos en el Valle de Aburrá (2021-2024)")
    st.markdown(
        "Este proyecto reúne mediciones de PM2.5 y registros de nacimientos en los municipios del Valle de Aburrá. "
        "Las visualizaciones permiten explorar la evolución temporal de PM2.5 y analizar indicadores perinatales "
        "como prematuridad, bajo peso y talla baja."
    )

    st.markdown("## Alcance")
    st.write(
        "- Medidas mensuales de PM2.5 por municipio.\n"
        "- Registros de nacimientos con clasificación de peso y talla.\n"
        "- Análisis de tasas (prematuridad, bajo peso, talla baja) y correlación con PM2.5 en gestación."
    )

    # === Sección de Equipo ===
    st.markdown("## 👥 Equipo responsable")
    equipo = [
        "Juan Alejandro Ruiz Gutiérrez",
        "Ana Yuleisi Palma Torres",
        "Oscar Javier Lara Guzmán",
        "Daniel Campillo Villa"
    ]
    
    cols_equipo = st.columns(len(equipo))
    for idx, miembro in enumerate(equipo):
        with cols_equipo[idx]:
            st.write(f"**{miembro}**")

    # === Sección de Fuentes y logos ===
    st.markdown("## Fuentes de datos")
    st.write(
        "DANE (Departamento Administrativo Nacional de Estadística): la autoridad nacional en estadísticas y "
        "registros vitales de Colombia.\n\n"
        "SIATA (Sistema de Alerta Temprana del Valle de Aburrá): sistema regional que monitorea la calidad del aire "
        "en Medellín y municipios del Valle de Aburrá."
    )

    # Preparar paths de placeholder (carpeta assets dentro de src)
    base = _base_path()
    dane_path = base / "assets" / "dane_logo.png"
    siata_path = base / "assets" / "siata_logo.png"

    col_dane, col_siata = st.columns(2)
    with col_dane:
        if dane_path.exists():
            st.image(str(dane_path), width=260)
        else:
            st.info("Placeholder imagen DANE\nColoque 'dane_logo.png' en src/assets para mostrar el logo.")

    with col_siata:
        if siata_path.exists():
            st.image(str(siata_path), width=200)
        else:
            st.info("Placeholder imagen SIATA\nColoque 'siata_logo.png' en src/assets para mostrar el logo.")