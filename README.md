# Lab 1 – Series de Tiempo: Viajeros Internacionales a Guatemala

**CC3084 – Data Science | Universidad del Valle de Guatemala**
**Semestre II 2026 | Grupo: 3 personas**

---

## Objetivo del Proyecto

Este repositorio contiene la resolución del **Laboratorio 1 de Series de Tiempo**. Analizamos el ingreso histórico de viajeros a Guatemala (2009–2026) para construir, diagnosticar y predecir series de tiempo utilizando técnicas estadísticas clásicas (ARIMA, Holt-Winters, Suavizamiento) y modernas (Prophet).

Para facilitar la redacción del informe, el código ha sido optimizado para generar **resultados altamente objetivos y consolidados**:
1. **Salida de texto limpia**: Los scripts imprimen conclusiones directas en la consola (p-values, parámetros sugeridos, y análisis comparativo) listas para copiar al PDF.
2. **Dashboards consolidados**: En lugar de decenas de imágenes sueltas, se genera un único "Dashboard" maestró por cada serie de tiempo, conteniendo todas las evidencias gráficas requeridas (Descomposición, ACF, PACF, Residuos).

## Estructura del Proyecto

```
time-series-lab/
├── README.md
├── requirements.txt
├── data/
│   └── raw/                    # Colocar el archivo .xlsx original aquí
├── src/                        # Lógica de negocio (limpieza y construcción)
│   ├── __init__.py
│   ├── data_loader.py          # Limpieza estandarizada
│   ├── series_builder.py       # Definición de series
│   ├── models.py               # Ajuste de modelos
│   └── metrics.py              # AIC, BIC, RMSE, MAE
├── analysis/                   # SCRIPTS PRINCIPALES PARA EJECUTAR
│   ├── 01_eda.py               # Exploración inicial (9 gráficas)
│   ├── 02_series_tiempo.py     # Diagnóstico de series (8 dashboards)
│   └── 03_modelos.py           # Predicciones y residuos (8 dashboards)
└── reports/
    └── figures/                # Aquí aparecerán los dashboards (.png)
```

## Instrucciones de Ejecución

Sigue estos pasos en tu terminal para replicar todo el análisis desde cero:

### 1. Clonar el repositorio y entrar a la carpeta
```bash
git clone https://github.com/GerardoFdez7/time-series-lab.git
cd time-series-lab
```

### 2. Crear entorno virtual (Recomendado)
```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Colocar el dataset
Copia el archivo Excel del laboratorio exactamente en esta ruta:
`data/raw/Base_Migracion_2009-2026jun.xlsx`

### 5. Correr los Análisis (Generación de Evidencia)

**Paso A: Análisis Exploratorio (EDA)**
```bash
python analysis/01_eda.py
```
*Genera las gráficas básicas en `reports/figures/eda_*.png`.*

**Paso B: Diagnóstico de Series de Tiempo**
```bash
python analysis/02_series_tiempo.py
```
*Genera un `dashboard_<serie>.png` por serie. **Lee la consola cuidadosamente**, imprime las conclusiones de estacionalidad (Dickey-Fuller) y la Sección 5 completa para el informe.*

**Paso C: Modelado y Predicción**
```bash
python analysis/03_modelos.py
```
*Toma unos 10-15 minutos. Genera un `dashboard_modelos_<serie>.png` comparando predicciones y mostrando residuos del mejor modelo. La consola imprimirá tablas comparativas de MAE y RMSE.*

## Categorías de Series Seleccionadas

1. **Serie obligatoria**: Total mensual de viajeros (Turista + Excursionista)
2. **Vías de ingreso**: Aérea, Terrestre, Marítima
3. **Tipo de viajero**: Turista, Excursionista, Crucerista, Viajero, Visitante

> **Nota sobre 2022-2023**: La categoría "Viajero" cae fuertemente en 2023 por exclusión de viajeros no turísticos. Para series comparables en todo el período usar Turista + Excursionista.
