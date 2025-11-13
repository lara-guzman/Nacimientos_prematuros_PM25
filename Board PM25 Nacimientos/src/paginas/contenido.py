from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.faker import Faker
from streamlit_echarts import st_pyecharts
import seaborn as sns
import unicodedata

def cargar_pm25():
    st.title("Datos de PM2.5 en Medellín")
    # Ejemplo de datos simulados o cargados desde un archivo CSV/local
    data = {
        "Año-Mes": ["2021-01", "2021-02", "2021-03"],
        "PM2.5 (µg/m³)": [30, 28, 29]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)
    st.line_chart(df.set_index("Año-Mes"))

def cargar_nacimientos():
    # Obtener la ruta base del archivo actual
    base_path = Path(__file__).resolve().parent.parent  # sube dos niveles: de paginas a src

    # Construir la ruta hacia el archivo CSV
    csv_path = base_path / 'datasets' / 'consolidado.csv'
    
    # === 1. Cargar archivo base ===
    df = pd.read_csv(csv_path, encoding="utf-8")

    # === 2. Mapas de categorías ===
    map_peso = {
        "Menos de 1.000 g": 1,
        "1.000 - 1.499 g": 2, 
        "1.500 - 1.999 g": 3,
        "2.000 - 2.499 g": 4,
        "2.500 - 2.999 g": 5,
        "3.000 - 3.499 g": 6,
        "3.500 - 3.999 g": 7,
        "4.000 g o más": 8
    }

    map_talla = {
        "Menos de 20 cm": 1,
        "20 - 29 cm": 2,
        "30 - 39 cm": 3,
        "40 - 49 cm": 4,
        "50 - 59 cm": 5,
        "60 cm o más": 6
    }

    map_tges = {
        "Menos de 22 semanas": 1,
        "22 a 27 semanas": 2,
        "28 a 36 semanas": 3,
        "37 a 41 semanas": 4,
        "42 o más semanas": 5
    }

    map_edad_madre = {
        "10-14 años": 12.0,
        "15-19 años": 17.0,
        "20-24 años": 22.0,
        "25-29 años": 27.0,
        "30-34 años": 32.0,
        "35-39 años": 37.0,
        "40-44 años": 42.0,
        "45-49 años": 47.0,
        "50-54 años": 52.0
    }

    # === 3. Crear variables numéricas y binarias ===
    df["peso_cat"] = df["PESO_NAC"].map(map_peso)
    df["talla_cat"] = df["TALLA_NAC"].map(map_talla)
    df["tges_cat"] = df["T_GES_AGRU_CIE"].map(map_tges)
    df["edad_madre"] = df["EDAD_MADRE"].map(map_edad_madre)

    df["prematuro"] = df["tges_cat"].apply(lambda x: 1 if x <= 3 else 0)
    df["bajo_peso"] = df["peso_cat"].apply(lambda x: 1 if x <= 4 else 0)
    df["talla_baja"] = df["talla_cat"].apply(lambda x: 1 if x <= 3 else 0)

    # === 4. Resumen general ===
    totales = {
        "Total_nacimientos": len(df),
        "Total_prematuros": df["prematuro"].sum(),
        "Total_bajo_peso": df["bajo_peso"].sum(),
        "Total_talla_baja": df["talla_baja"].sum()
    }

    tasas = {
        "Tasa_prematuridad_%": 100 * df["prematuro"].mean(),
        "Tasa_bajo_peso_%": 100 * df["bajo_peso"].mean(),
        "Tasa_talla_baja_%": 100 * df["talla_baja"].mean()
    }

    # === 5. Resumen por municipio y año ===
    df["ANO"] = df["ANO"].astype(int)

    resumen = (
        df.groupby(["CODMUNRE", "ANO"])
        .agg(
            Total_nacimientos=("prematuro", "count"),
            Prematuros=("prematuro", "sum"),
            Bajo_peso=("bajo_peso", "sum"),
            Talla_baja=("talla_baja", "sum")
        )
        .reset_index()
    )

    # Tasas porcentuales
    resumen["Tasa_prematuros_%"] = 100 * resumen["Prematuros"] / resumen["Total_nacimientos"]
    resumen["Tasa_bajo_peso_%"] = 100 * resumen["Bajo_peso"] / resumen["Total_nacimientos"]
    resumen["Tasa_talla_baja_%"] = 100 * resumen["Talla_baja"] / resumen["Total_nacimientos"]

    # === 6. Ranking global por municipio ===
    ranking = (
        resumen.groupby("CODMUNRE")[["Tasa_prematuros_%", "Tasa_bajo_peso_%", "Tasa_talla_baja_%"]]
        .mean()
        .sort_values("Tasa_prematuros_%", ascending=False)
    )

    # === 7. Visualizaciones ===
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Gráfico 1: Tasas promedio por municipio ---
    ranking.plot(kind="bar", figsize=(10,6))
    plt.title("Promedio de tasas por municipio (2021–2024)")
    plt.ylabel("Porcentaje (%)")
    plt.xlabel("Municipio")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt)

    # Analisis de correlacion
    # Calcular matriz de correlación
    correlacion = df[["prematuro", "bajo_peso", "talla_baja", "edad_madre", "pm25_gestation_avg"]].corr(method="pearson")

    # Cambiar etiquetas para mostrar nombres más legibles
    etiquetas = {
        "prematuro": "Prematuro",
        "bajo_peso": "Bajo Peso",
        "talla_baja": "Baja Talla",
        "edad_madre": "Edad Madre",
        "pm25_gestation_avg": "PM25"
    }

    correlacion.rename(index=etiquetas, columns=etiquetas, inplace=True)

    # Mostrar matriz de correlación como texto
    st.write("Matriz de correlación entre prematuro, bajo peso y talla baja:")
    st.dataframe(correlacion)

    # Crear mapa de calor
    plt.figure(figsize=(6,4))
    sns.heatmap(correlacion, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Mapa de calor de correlación")
    st.pyplot(plt)
    plt.clf()

    # --- Gráfico 2: Evolución anual ---
    tasas_anuales = (
        resumen.groupby("ANO")[["Tasa_prematuros_%", "Tasa_bajo_peso_%", "Tasa_talla_baja_%"]]
        .mean()
    )

    tasas_anuales.plot(marker="o", figsize=(8,5))
    plt.title("Evolución anual de tasas (Valle de Aburrá, 2021–2024)")
    plt.ylabel("Porcentaje (%)")
    plt.xlabel("Año")
    plt.legend(title="Indicador")
    plt.tight_layout()
    st.pyplot(plt)

    # Agrupar total de nacimientos por municipio
    total_nacimientos_por_mun = resumen.groupby("CODMUNRE")["Total_nacimientos"].sum()

    # Graficar torta
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        total_nacimientos_por_mun,
        labels=total_nacimientos_por_mun.index,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 9}
    )

    fig = _grafico_torta_con_etiquetas_fuera(total_nacimientos_por_mun, total_nacimientos_por_mun.index, "Distribución porcentual de nacimientos por municipio (2021-2024)", "Municipios")
    st.pyplot(plt)

    # Mover los porcentajes fuera de la torta
    

    # Agrupar suma de nacimientos prematuros por municipio
    prematuros_por_mun = resumen.groupby("CODMUNRE")["Prematuros"].sum()

    # Graficar anillo (donut) usando la función helper
    fig_donut_prematuros = _grafico_torta_con_etiquetas_fuera(
        prematuros_por_mun.values,
        prematuros_por_mun.index,
        "Distribución porcentual de nacimientos prematuros por municipio (2021-2024)",
        "Municipios"
    )
    st.pyplot(fig_donut_prematuros)
    plt.clf()

    # --- Gráfico 2: Evolución anual ---
    tasas_anuales = (
        resumen.groupby("ANO")[["Tasa_prematuros_%", "Tasa_bajo_peso_%", "Tasa_talla_baja_%"]]
        .mean()
    )

    tasas_anuales.plot(marker="o", figsize=(8,5))
    plt.title("Evolución anual de tasas (Valle de Aburrá, 2021–2024)")
    plt.ylabel("Porcentaje (%)")
    plt.xlabel("Año")
    plt.legend(title="Indicador")
    plt.tight_layout()
    st.pyplot(plt)

    # --- Gráfico nuevo: Casos de peso bajo por año ---
    peso_bajo_por_año = df.groupby("ANO")["bajo_peso"].sum().sort_index()
    plt.figure(figsize=(8,4.5))
    x = peso_bajo_por_año.index.astype(int)
    y = peso_bajo_por_año.values
    bars = plt.bar(x, y, color='tab:orange', edgecolor='black')
    plt.title("Casos de peso bajo por año (Valle de Aburrá, 2021–2024)")
    plt.xlabel("Año")
    plt.ylabel("Número de casos de peso bajo")
    plt.xticks(x)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Añadir etiquetas con los valores encima de cada barra
    max_y = y.max() if len(y) > 0 else 0
    offset = max_y * 0.01 if max_y > 0 else 0.1
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            f"{int(h)}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def _grafico_torta_con_etiquetas_fuera(valores, labels, titulo, legend_title):
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        valores,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11},     # porcentaje: un poco más grande
        pctdistance=0.80,
        wedgeprops=dict(width=0.38, edgecolor='white', linewidth=2)  # Crea el anillo
    )
    
    # Ajustar estilo de los porcentajes dentro del anillo
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')
    
    plt.setp(autotexts, size=11, weight="bold")
    # Leyenda con etiquetas de municipios más grandes
    lgd = ax.legend(wedges, labels, title=legend_title, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12)
    # Asegurar que todos los textos de la leyenda tengan el mismo tamaño
    for t in lgd.get_texts():
        t.set_fontsize(12)
    if lgd.get_title():
        lgd.get_title().set_fontsize(12)
    
    plt.title(titulo, fontsize=13, weight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    return fig

def cargar_pm25():
    # Obtener la ruta base del archivo actual
    base_path = Path(__file__).resolve().parent.parent  # sube dos niveles: de paginas a src

    # Construir la ruta hacia el archivo CSV
    csv_path = base_path / 'datasets' / 'pm25_mensual_municipio_202511071445.csv'
    df = pd.read_csv(csv_path)
    df['anio_mes'] = pd.to_datetime(df['anio_mes'])  # Asegura que la columna sea de tipo fecha

    # Checkbox múltiple para seleccionar municipios
    municipios_disponibles = sorted(df['municipio'].unique())

    # Añadir “Todos” en el checklist
    opciones = ['Todos'] + municipios_disponibles
    municipios_seleccionados = st.multiselect(
        'Selecciona uno o más municipios para observar',
        opciones,
        default=['Medellín', 'Itagüí', 'Envigado', 'Sabaneta', 'Bello']
    )

    # Si “Todos” está seleccionado, mostrar todos los municipios
    if 'Todos' in municipios_seleccionados:
        municipios_plot = municipios_disponibles
    else:
        municipios_plot = municipios_seleccionados

    plt.figure(figsize=(14, 5))  # Aumenta el ancho de la gráfica
    for municipio in municipios_plot:
        datos = df[df['municipio'] == municipio].sort_values('anio_mes')
        plt.plot(datos['anio_mes'], datos['pm25'], marker='o', label=municipio)

    plt.title('Evolución mensual de PM2.5 por municipio')
    plt.xlabel('Fecha')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

    plt.figure()
    ranking = df.groupby("municipio")["pm25"].mean().sort_values()
    ranking.plot(kind='barh', figsize=(10,6), color='skyblue')
    plt.xlabel('PM2.5 promedio anual')
    plt.title('Ranking anual de PM2.5 por municipio (2021–2024)')
    st.pyplot(plt)

    plt.figure(figsize=(14,6))
    sns.boxplot(data=df, x='municipio', y='pm25')
    plt.title('Distribución mensual de PM2.5 por municipio')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)


