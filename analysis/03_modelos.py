"""
analysis/03_modelos.py
Lab 1 – Series de Tiempo | CC3084 Data Science
Persona 3: Modelos de predicción y comparación

Ejecutar desde la raíz del proyecto:
    python analysis/03_modelos.py

Los gráficos se guardan en reports/figures/modelos_*.png

MODELOS:
  - Múltiples ARIMA (manual)
  - Auto-ARIMA (pmdarima)
  - Holt-Winters
  - Suavizamiento Exponencial
  - Seasonal Naïve
  - Prophet (Meta)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.data_loader import get_clean_data
from src.series_builder import (
    build_total_series,
    build_via_series,
    build_tipo_viajero_series,
)
from src.models import (
    fit_arima,
    fit_auto_arima,
    fit_holt_winters,
    fit_simple_exp_smoothing,
    fit_seasonal_naive,
    fit_prophet,
)
from src.metrics import print_comparison_table

# ------------------------------------------------------------------ #
# Configuración                                                         #
# ------------------------------------------------------------------ #

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

PALETTE = sns.color_palette("tab10")
TRAIN_RATIO = 0.70


def save_fig(name: str) -> None:
    path = FIGURES_DIR / f"modelos_{name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=120)
    print(f"  → Guardado: {path.name}")
    plt.close()


def split_train_test(serie: pd.Series, ratio: float = TRAIN_RATIO):
    n = int(len(serie) * ratio)
    return serie.iloc[:n], serie.iloc[n:]


# ================================================================== #
# ANÁLISIS DE RESIDUOS                                                  #
# ================================================================== #


def analizar_residuos(fit_result: dict, serie_nombre: str) -> None:
    """
    Analiza los residuos de un modelo ARIMA ajustado.
    Requiere que fit_result['fit'] sea un objeto statsmodels SARIMAX.
    """
    if fit_result.get("fit") is None:
        print("  Análisis de residuos no disponible para este modelo.")
        return

    try:
        fit = fit_result["fit"]
        residuos = fit.resid

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        # 1. Residuos en el tiempo
        ax = axes[0, 0]
        ax.plot(residuos.index, residuos.values, color=PALETTE[0], linewidth=1)
        ax.axhline(0, color="red", linewidth=0.8)
        ax.set_title("Residuos en el tiempo")

        # 2. Histograma de residuos
        ax = axes[0, 1]
        ax.hist(residuos.dropna(), bins=30, color=PALETTE[1], edgecolor="white", density=True)
        from scipy import stats
        mu, std = stats.norm.fit(residuos.dropna())
        x = np.linspace(residuos.min(), residuos.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, std), color="red", linewidth=2)
        ax.set_title("Distribución de residuos")

        # 3. Q-Q Plot
        ax = axes[1, 0]
        stats.probplot(residuos.dropna(), plot=ax)
        ax.set_title("Q-Q Plot de residuos")

        # 4. ACF de residuos
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuos.dropna(), lags=20, ax=axes[1, 1])
        axes[1, 1].set_title("ACF de residuos")

        modelo_name = fit_result["modelo"].replace("(", "_").replace(")", "").replace(",", "_")
        serie_safe = serie_nombre.replace(" ", "_").replace(":", "")
        fig.suptitle(f"Análisis de Residuos: {fit_result['modelo']} | {serie_nombre}",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        save_fig(f"residuos_{serie_safe}_{modelo_name}")

    except Exception as e:
        print(f"  ADVERTENCIA en análisis de residuos: {e}")


# ================================================================== #
# GRÁFICO DE PREDICCIONES vs REALES                                     #
# ================================================================== #


def plot_predicciones(
    train: pd.Series,
    test: pd.Series,
    resultados_modelos: list[dict],
    serie_nombre: str,
) -> None:
    """Grafica las predicciones de todos los modelos vs los valores reales."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Últimos 24 meses de train para contexto
    train_tail = train.iloc[-24:]
    ax.plot(train_tail.index, train_tail.values / 1e3,
            color="gray", linewidth=1.5, label="Train (últimos 24m)", alpha=0.6)
    ax.plot(test.index, test.values / 1e3,
            color="black", linewidth=2, label="Real (test)", linestyle="--")

    colors = PALETTE[1:]
    for i, r in enumerate(resultados_modelos):
        pred = r["predicciones"]
        ax.plot(pred.index, pred.values / 1e3,
                color=colors[i % len(colors)], linewidth=1.5,
                label=r["modelo"], alpha=0.8)

    ax.set_title(f"Predicciones vs Reales: {serie_nombre}", fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Miles de viajeros")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}K"))
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()

    safe = serie_nombre.replace(" ", "_").replace(":", "")
    save_fig(f"predicciones_{safe}")


# ================================================================== #
# ANÁLISIS COMPLETO DE UNA SERIE                                        #
# ================================================================== #


def modelar_serie(
    info,
    p: int = 1, d: int = 1, q: int = 1,
    run_prophet: bool = True,
    run_auto_arima: bool = True,
) -> list[dict]:
    """
    Ajusta todos los modelos requeridos para una serie y compara resultados.

    Parameters
    ----------
    info : SerieInfo
        Metadata + serie de tiempo.
    p, d, q : int
        Parámetros ARIMA base (punto de partida; se prueban variaciones).
    run_prophet : bool
        Si True, ajusta Prophet (puede ser lento).
    run_auto_arima : bool
        Si True, ejecuta auto_arima.

    Returns
    -------
    list[dict]
        Lista de resultados de todos los modelos.
    """
    serie = info.serie
    nombre = info.nombre
    train, test = split_train_test(serie)

    print("\n" + "━" * 60)
    print(f"MODELANDO: {nombre}")
    print(f"  Train: {len(train)} obs | Test: {len(test)} obs")
    print("━" * 60)

    resultados = []

    # ── g) Múltiples ARIMA ───────────────────────────────────────
    print("\n── Modelos ARIMA (múltiples configuraciones) ──")
    arima_configs = [
        (p, d, q),
        (p + 1, d, q),
        (p, d, q + 1),
        (2, d, 2),
        (0, d, 1),   # MA puro
        (1, d, 0),   # AR puro
    ]

    # Eliminar duplicados
    arima_configs = list(dict.fromkeys(arima_configs))

    for config in arima_configs:
        try:
            r = fit_arima(train, test, order=config)
            resultados.append(r)
        except Exception as e:
            print(f"  ARIMA{config} falló: {e}")

    # SARIMA estacional (12 meses)
    try:
        r = fit_arima(
            train, test,
            order=(p, d, q),
            seasonal_order=(1, 1, 1, 12),
            model_name=f"SARIMA({p},{d},{q})(1,1,1)[12]",
        )
        resultados.append(r)
    except Exception as e:
        print(f"  SARIMA falló: {e}")

    # ── f) Auto-ARIMA ────────────────────────────────────────────
    if run_auto_arima:
        try:
            print("\n── Auto-ARIMA ──")
            r = fit_auto_arima(train, test)
            resultados.append(r)
        except Exception as e:
            print(f"  auto_arima falló: {e}")

    # ── h) Holt-Winters ──────────────────────────────────────────
    print("\n── Holt-Winters ──")
    for trend in ["add", "mul"]:
        for seasonal in ["add", "mul"]:
            try:
                r = fit_holt_winters(train, test, trend=trend, seasonal=seasonal)
                resultados.append(r)
            except Exception as e:
                print(f"  HW(trend={trend}, seasonal={seasonal}) falló: {e}")

    # ── Suavizamiento Exponencial Simple ─────────────────────────
    try:
        r = fit_simple_exp_smoothing(train, test)
        resultados.append(r)
    except Exception as e:
        print(f"  Suavizamiento exponencial falló: {e}")

    # ── Seasonal Naïve ───────────────────────────────────────────
    try:
        r = fit_seasonal_naive(train, test)
        resultados.append(r)
    except Exception as e:
        print(f"  Seasonal Naïve falló: {e}")

    # ── Prophet ──────────────────────────────────────────────────
    if run_prophet:
        try:
            print("\n── Prophet ──")
            r = fit_prophet(train, test)
            resultados.append(r)
        except Exception as e:
            print(f"  Prophet falló: {e}")

    # ── Análisis de residuos del mejor ARIMA ─────────────────────
    arima_results = [r for r in resultados if "ARIMA" in r["modelo"] or "SARIMA" in r["modelo"]]
    if arima_results:
        best_arima = min(
            arima_results,
            key=lambda r: r["metricas"]["MAE"]
        )
        print(f"\n── Análisis de residuos: {best_arima['modelo']} ──")
        analizar_residuos(best_arima, nombre)

    # ── j/k) Tabla comparativa de métricas ───────────────────────
    metricas_list = [r["metricas"] for r in resultados]
    print_comparison_table(metricas_list)

    # ── i) Gráfico de predicciones ────────────────────────────────
    # Graficar solo los 5 mejores modelos por MAE para no saturar
    resultados_ordenados = sorted(resultados, key=lambda r: r["metricas"]["MAE"])
    top_5 = resultados_ordenados[:5]
    plot_predicciones(train, test, top_5, nombre)

    # Mejor modelo global
    mejor = resultados_ordenados[0]
    print(f"\n✓ MEJOR MODELO para '{nombre}': {mejor['modelo']}")
    print(f"  MAE={mejor['metricas']['MAE']:,.0f} | RMSE={mejor['metricas']['RMSE']:,.0f}")

    return resultados


# ================================================================== #
# MAIN                                                                  #
# ================================================================== #


def main() -> None:
    print("=" * 60)
    print("MODELOS DE PREDICCIÓN – Viajeros Internacionales")
    print("Lab 1 | CC3084 Data Science | UVG 2026")
    print("=" * 60)

    df = get_clean_data()

    # ── Serie Total ──────────────────────────────────────────────
    print("\n\n══════════════════════════════════════════════")
    print("SERIE 1: TOTAL MENSUAL (OBLIGATORIA)")
    print("══════════════════════════════════════════════")
    total_info = build_total_series(df)
    # p=1, d=1, q=1 como punto de partida (ajustar con resultados de 02_series_tiempo.py)
    modelar_serie(total_info, p=1, d=1, q=1)

    # ── Series por Vía ───────────────────────────────────────────
    print("\n\n══════════════════════════════════════════════")
    print("SERIES POR VÍA DE INGRESO")
    print("══════════════════════════════════════════════")
    via_series = build_via_series(df)
    for via_name, via_info in via_series.items():
        modelar_serie(via_info, p=1, d=1, q=1, run_prophet=True)

    # ── Series por Tipo de Viajero ───────────────────────────────
    print("\n\n══════════════════════════════════════════════")
    print("SERIES POR TIPO DE VIAJERO")
    print("══════════════════════════════════════════════")
    tipo_series = build_tipo_viajero_series(df)
    for tipo_name, tipo_info in tipo_series.items():
        # Para tipos con pocos datos, omitir Prophet para velocidad
        n_obs = len(tipo_info.serie)
        modelar_serie(
            tipo_info, p=1, d=1, q=1,
            run_prophet=(n_obs >= 60),
            run_auto_arima=(n_obs >= 24),
        )

    print("\n" + "=" * 60)
    print(f"Modelos completados. Figuras en: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
