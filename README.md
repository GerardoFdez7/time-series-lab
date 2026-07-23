# Lab 1 – Series de Tiempo: Viajeros Internacionales a Guatemala

**CC3084 – Data Science | Universidad del Valle de Guatemala**
**Semestre II 2026 | Grupo: 3 personas**

---

## Descripción

Análisis de series de tiempo sobre datos históricos de ingreso de viajeros internacionales a Guatemala (2009–2026). Incluye análisis exploratorio, construcción de series mensuales, modelado ARIMA/Prophet/Holt-Winters y predicción.

## Estructura del Proyecto

```
lab2-data-science/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   └── raw/                    # Colocar el .xlsx aquí (no se versiona)
├── src/                        # Módulos compartidos (no modificar sin coordinar)
│   ├── __init__.py
│   ├── data_loader.py          # Carga y limpieza del dataset
│   ├── series_builder.py       # Construcción de series de tiempo
│   ├── models.py               # Wrappers de modelos (ARIMA, Prophet, etc.)
│   └── metrics.py              # Cálculo de métricas (MAE, RMSE, AIC, BIC)
├── analysis/                   # Scripts de análisis (uno por persona)
│   ├── 01_eda.py               # Persona 1: Análisis Exploratorio
│   ├── 02_series_tiempo.py     # Persona 2: Creación y análisis de series
│   └── 03_modelos.py           # Persona 3: Modelos y predicciones
└── reports/
    └── figures/                # Gráficos generados automáticamente
```

## Setup

### 1. Clonar el repositorio
```bash
git clone <URL_DEL_REPO>
```

### 2. Crear entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows
```

### Compatibilidad de Python

- Recomendado: Python 3.12 (máxima compatibilidad, incluye `pmdarima` y `prophet`).
- Python 3.13: el proyecto corre, pero `pmdarima` y `prophet` se omiten automáticamente por falta de soporte estable.

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 5. Ejecutar los análisis
```bash
# EDA completo
python analysis/01_eda.py

# Series de tiempo
python analysis/02_series_tiempo.py

# Modelos y predicciones
python analysis/03_modelos.py
```

Los gráficos se guardan automáticamente en `reports/figures/`.

## Categorías de Series Seleccionadas

1. **Serie obligatoria**: Total mensual de viajeros (Turista + Excursionista)
2. **Vías de ingreso**: Aérea, Terrestre, Marítima
3. **Tipo de viajero**: Turista, Excursionista, Crucerista, Viajero, Visitante

> **Nota sobre 2022-2023**: La categoría "Viajero" cae fuertemente en 2023 por exclusión de viajeros no turísticos. Para series comparables en todo el período usar Turista + Excursionista.
