"""
analysis/02_series_tiempo.py
Lab 1 – Series de Tiempo | CC3084 Data Science
Persona 2: Construcción y análisis de series de tiempo

Ejecutar desde la raíz del proyecto:
    python analysis/02_series_tiempo.py

Los gráficos se guardan en reports/figures/series_*.png

SERIES ANALIZADAS:
  1. Total mensual (Turista + Excursionista) — obligatoria
  2. Por vía de ingreso: Aérea, Terrestre, Marítima
  3. Por tipo de viajero: Turista, Excursionista, etc.
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

# Series de tiempo
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.data_loader import get_clean_data
from src.series_builder import (
    build_total_series,
    build_via_series,
    build_tipo_viajero_series,
    build_all_series,
    print_series_summary,
    SerieInfo,
)

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
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

PALETTE = sns.color_palette("tab10")

TRAIN_RATIO = 0.70   # 70% entrenamiento, 30% prueba


def save_fig(name: str) -> None:
    path = FIGURES_DIR / f"series_{name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=120)
    print(f"  → Guardado: {path.name}")
    plt.close()


# ================================================================== #
# DIVISIÓN TRAIN / TEST                                                 #
# ================================================================== #


def split_train_test(serie: pd.Series, train_ratio: float = TRAIN_RATIO):
    """
    Divide una serie temporal en entrenamiento y prueba.
    Mantiene el orden temporal (no aleatorio).

    Parameters
    ----------
    serie : pd.Series
        Serie con DatetimeIndex mensual.
    train_ratio : float
        Proporción de datos para entrenamiento (0.70 = 70%).

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (train, test)
    """
    n = len(serie)
    n_train = int(n * train_ratio)
    train = serie.iloc[:n_train]
    test = serie.iloc[n_train:]
    return train, test


# ================================================================== #
# ANÁLISIS INDIVIDUAL DE UNA SERIE                                      #
# ================================================================== #


def analizar_serie(info: SerieInfo, color=None, prefix: str = "") -> dict:
    """
    Realiza el análisis completo de una serie de tiempo:
      a) Metadata (inicio, fin, frecuencia)
      b) Gráfico de la serie
      c) Descomposición (tendencia, estacionalidad, residuo)
      d) Estacionariedad en media y varianza
      e) ACF y PACF
      f) Prueba ADF (Dickey-Fuller Aumentada)

    Parameters
    ----------
    info : SerieInfo
        Metadata + serie de tiempo.
    color : color matplotlib
        Color para el gráfico principal.
    prefix : str
        Prefijo para los nombres de archivo de figuras.

    Returns
    -------
    dict
        Resultados del análisis (ADF, parámetros sugeridos p/d/q).
    """
    serie = info.serie
    nombre = info.nombre
    safe_name = prefix + nombre.replace(" ", "_").replace(":", "").replace("/", "_")

    print("\n" + "━" * 60)
    print(f"SERIE: {nombre}")
    print("━" * 60)

    # ── a) Metadata ──────────────────────────────────────────────
    print(f"  Inicio     : {info.inicio}")
    print(f"  Fin        : {info.fin}")
    print(f"  Frecuencia : {info.frecuencia}")
    print(f"  Obs. totales: {len(serie)}")
    print(f"  Descripción: {info.descripcion}")

    # División train/test
    train, test = split_train_test(serie)
    print(f"  Train      : {train.index[0].date()} → {train.index[-1].date()} ({len(train)} obs)")
    print(f"  Test       : {test.index[0].date()} → {test.index[-1].date()} ({len(test)} obs)")

    # ── b) Gráfico de la serie ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(train.index, train.values / 1e3, label="Entrenamiento", color=color or PALETTE[0], linewidth=1.5)
    ax.plot(test.index, test.values / 1e3, label="Prueba", color="tomato", linewidth=1.5)
    ax.axvspan(
        pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-01"),
        alpha=0.15, color="red", label="COVID-19"
    )
    ax.set_title(f"Serie: {nombre}", fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Miles de viajeros")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}K"))
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(f"{safe_name}_a_grafico")

    # ── c) Descomposición ────────────────────────────────────────
    # Usar solo datos de entrenamiento para la descomposición
    # Necesitamos al menos 2 ciclos completos para descomposición (24 obs)
    if len(train) >= 24:
        try:
            decomp = seasonal_decompose(train, model="additive", period=12)

            fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
            axes[0].plot(train.index, decomp.observed / 1e3, color=color or PALETTE[0])
            axes[0].set_ylabel("Original (K)")
            axes[1].plot(train.index, decomp.trend / 1e3, color=PALETTE[1])
            axes[1].set_ylabel("Tendencia (K)")
            axes[2].plot(train.index, decomp.seasonal / 1e3, color=PALETTE[2])
            axes[2].set_ylabel("Estacional (K)")
            axes[3].plot(train.index, decomp.resid / 1e3, color=PALETTE[3])
            axes[3].axhline(0, color="black", linewidth=0.8)
            axes[3].set_ylabel("Residuo (K)")

            for ax in axes:
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}K"))

            fig.suptitle(f"Descomposición Aditiva: {nombre}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            save_fig(f"{safe_name}_b_descomposicion")

            # Magnitud de la estacionalidad vs tendencia
            seasonal_strength = decomp.seasonal.std()
            trend_strength = decomp.trend.dropna().std()
            print(f"\n  Fuerza estacional (std): {seasonal_strength:,.0f}")
            print(f"  Fuerza de tendencia (std): {trend_strength:,.0f}")
        except Exception as e:
            print(f"  ADVERTENCIA descomposición: {e}")
    else:
        print("  Descomposición omitida (menos de 24 observaciones en train).")

    # ── d) Estacionariedad en varianza (Coef. de Variación) ─────
    print("\n  ── Estacionariedad en varianza ──")
    rolling_std = train.rolling(window=12).std()
    rolling_mean = train.rolling(window=12).mean()
    cv_series = (rolling_std / rolling_mean).dropna()
    cv_overall = train.std() / train.mean()
    print(f"  Coef. Variación global: {cv_overall:.4f}")
    print(f"  CV rodante (media): {cv_series.mean():.4f}")

    needs_log = cv_series.std() > 0.15
    if needs_log:
        print("  → Se recomienda transformación logarítmica (varianza no constante)")
    else:
        print("  → Varianza relativamente estable. Transformación logarítmica opcional.")

    # Gráfico: rolling mean & std
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axes[0].plot(train.index, train.values / 1e3, color=color or PALETTE[0], alpha=0.7, label="Serie")
    axes[0].plot(rolling_mean.index, rolling_mean.values / 1e3, color="red", label="Media móvil (12m)")
    axes[0].set_ylabel("Miles de viajeros")
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"Media y Desviación Estándar Rodante: {nombre}", fontweight="bold")

    axes[1].plot(rolling_std.index, rolling_std.values / 1e3, color=PALETTE[3], label="Desv. Estándar móvil (12m)")
    axes[1].set_ylabel("Std (miles)")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    save_fig(f"{safe_name}_c_rolling_stats")

    # ── e) Prueba ADF (Dickey-Fuller Aumentada) ──────────────────
    print("\n  ── Prueba ADF (Dickey-Fuller Aumentada) ──")
    adf_result = adfuller(train.dropna(), autolag="AIC")
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    n_lags = adf_result[2]
    critical_values = adf_result[4]

    print(f"  Estadístico ADF : {adf_stat:.6f}")
    print(f"  p-value         : {p_value:.6f}")
    print(f"  Nro. de lags    : {n_lags}")
    print("  Valores críticos:")
    for level, val in critical_values.items():
        print(f"    {level}: {val:.6f}")

    is_stationary = p_value < 0.05
    if is_stationary:
        print("  → CONCLUSIÓN: La serie ES estacionaria (p < 0.05). d = 0")
        d_suggested = 0
    else:
        print("  → CONCLUSIÓN: La serie NO ES estacionaria (p ≥ 0.05). Se requiere diferenciación.")
        # ADF sobre serie diferenciada
        train_diff = train.diff().dropna()
        adf_diff = adfuller(train_diff, autolag="AIC")
        print(f"\n  ADF sobre serie diferenciada (d=1):")
        print(f"    Estadístico: {adf_diff[0]:.6f}, p-value: {adf_diff[1]:.6f}")
        if adf_diff[1] < 0.05:
            print("  → Una diferenciación (d=1) es suficiente.")
            d_suggested = 1
        else:
            print("  → Puede requerirse d=2. Verificar manualmente.")
            d_suggested = 2

    # ── e-i) ACF y PACF ─────────────────────────────────────────
    print("\n  ── ACF y PACF ──")
    n_lags_plot = min(36, len(train) // 2 - 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf(train.dropna(), lags=n_lags_plot, ax=axes[0], color=color or PALETTE[0])
    axes[0].set_title(f"ACF: {nombre}")

    plot_pacf(train.dropna(), lags=n_lags_plot, ax=axes[1], method="ywm",
              color=color or PALETTE[0])
    axes[1].set_title(f"PACF: {nombre}")

    plt.suptitle(f"Autocorrelación: {nombre}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(f"{safe_name}_d_acf_pacf")

    # Sugerir p y q basado en ACF/PACF
    acf_vals = acf(train.dropna(), nlags=n_lags_plot)
    pacf_vals = pacf(train.dropna(), nlags=n_lags_plot, method="ywm")

    # Límite de significancia (aprox. 1.96/sqrt(n))
    ci = 1.96 / np.sqrt(len(train))

    # p: número de lags PACF significativos (antes del primero no significativo)
    p_suggested = 0
    for i in range(1, min(6, len(pacf_vals))):
        if abs(pacf_vals[i]) > ci:
            p_suggested = i
        else:
            break

    # q: número de lags ACF significativos
    q_suggested = 0
    for i in range(1, min(6, len(acf_vals))):
        if abs(acf_vals[i]) > ci:
            q_suggested = i
        else:
            break

    print(f"\n  ── Parámetros ARIMA sugeridos (análisis preliminar) ──")
    print(f"  p (PACF): {p_suggested}")
    print(f"  d (ADF) : {d_suggested}")
    print(f"  q (ACF) : {q_suggested}")
    print(f"  ARIMA({p_suggested},{d_suggested},{q_suggested}) como punto de partida")
    print(f"  Nota: Verificar con auto_arima y comparar múltiples modelos en 03_modelos.py")

    # ACF sobre serie diferenciada (si aplica)
    if not is_stationary:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        train_diff = train.diff().dropna()
        plot_acf(train_diff, lags=n_lags_plot, ax=axes[0])
        axes[0].set_title(f"ACF (Diferenciada d=1): {nombre}")
        plot_pacf(train_diff, lags=n_lags_plot, ax=axes[1], method="ywm")
        axes[1].set_title(f"PACF (Diferenciada d=1): {nombre}")
        plt.suptitle(f"ACF/PACF Serie Diferenciada: {nombre}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        save_fig(f"{safe_name}_e_acf_pacf_diff")

    return {
        "nombre": nombre,
        "inicio": info.inicio,
        "fin": info.fin,
        "n_obs": len(serie),
        "n_train": len(train),
        "n_test": len(test),
        "adf_stat": adf_stat,
        "adf_pvalue": p_value,
        "estacionaria": is_stationary,
        "p_sugerido": p_suggested,
        "d_sugerido": d_suggested,
        "q_sugerido": q_suggested,
        "cv_global": cv_overall,
        "necesita_log": needs_log,
        "train": train,
        "test": test,
    }


# ================================================================== #
# RESUMEN COMPARATIVO DE SERIES                                         #
# ================================================================== #


def comparar_series(resultados: list[dict]) -> None:
    """Genera tabla y gráficos comparativos entre todas las series."""
    print("\n" + "=" * 60)
    print("RESUMEN COMPARATIVO DE SERIES")
    print("=" * 60)

    # Tabla resumen
    cols = ["nombre", "inicio", "fin", "n_obs", "adf_pvalue", "estacionaria",
            "p_sugerido", "d_sugerido", "q_sugerido", "cv_global"]
    df_res = pd.DataFrame(resultados)[cols]
    df_res["estacionaria"] = df_res["estacionaria"].map({True: "Sí", False: "No"})
    df_res["adf_pvalue"] = df_res["adf_pvalue"].round(4)
    df_res["cv_global"] = df_res["cv_global"].round(4)
    print(df_res.to_string(index=False))

    # Gráfico: todas las series en un solo gráfico (normalizadas)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, r in enumerate(resultados):
        serie_norm = r["train"] / r["train"].max()  # Normalizar 0-1 para comparar
        ax.plot(serie_norm.index, serie_norm.values,
                label=r["nombre"][:25], color=PALETTE[i % len(PALETTE)],
                linewidth=1.3, alpha=0.85)

    ax.axvspan(
        pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-01"),
        alpha=0.12, color="red", label="COVID-19"
    )
    ax.set_title("Comparación de Series (normalizadas 0-1)", fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor normalizado")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    save_fig("comparativa_todas_series")

    # Preguntas del análisis comparativo (para el informe)
    print("\n── Para el análisis comparativo (Sección 5 del lab) ──")
    # Mayor estacionalidad: CV más alto sugiere más variación estacional
    max_cv = max(resultados, key=lambda r: r["cv_global"])
    print(f"  Mayor volatilidad (CV): {max_cv['nombre']} (CV={max_cv['cv_global']:.4f})")


# ================================================================== #
# MAIN                                                                  #
# ================================================================== #


def main() -> None:
    print("=" * 60)
    print("ANÁLISIS DE SERIES DE TIEMPO – Viajeros Internacionales")
    print("Lab 1 | CC3084 Data Science | UVG 2026")
    print("=" * 60)

    # Cargar datos
    df = get_clean_data()

    # Construir todas las series
    print("\n── Construyendo series de tiempo ──")
    total_info = build_total_series(df)
    via_series = build_via_series(df)
    tipo_series = build_tipo_viajero_series(df)

    # Mostrar resumen de todas las series
    all_series = {total_info.nombre: total_info}
    all_series.update({k: v for k, v in via_series.items()})
    all_series.update({k: v for k, v in tipo_series.items()})
    print_series_summary(all_series)

    # ── Analizar Serie Total (Obligatoria) ───────────────────────
    resultados = []
    print("\n\n══════════════════════════════════════════════════════")
    print("CATEGORÍA 1: SERIE TOTAL (OBLIGATORIA)")
    print("══════════════════════════════════════════════════════")
    r = analizar_serie(total_info, color=PALETTE[0], prefix="total_")
    resultados.append(r)

    # ── Analizar Series por Vía de Ingreso ──────────────────────
    print("\n\n══════════════════════════════════════════════════════")
    print("CATEGORÍA 2: VÍAS DE INGRESO")
    print("══════════════════════════════════════════════════════")
    via_colors = [PALETTE[1], PALETTE[2], PALETTE[3]]
    for i, (via_name, via_info) in enumerate(via_series.items()):
        r = analizar_serie(via_info, color=via_colors[i % len(via_colors)], prefix="via_")
        resultados.append(r)

    # ── Analizar Series por Tipo de Viajero ─────────────────────
    print("\n\n══════════════════════════════════════════════════════")
    print("CATEGORÍA 3: TIPO DE VIAJERO")
    print("══════════════════════════════════════════════════════")
    tipo_colors = [PALETTE[4], PALETTE[5], PALETTE[6], PALETTE[7], PALETTE[8]]
    for i, (tipo_name, tipo_info) in enumerate(tipo_series.items()):
        r = analizar_serie(tipo_info, color=tipo_colors[i % len(tipo_colors)], prefix="tipo_")
        resultados.append(r)

    # ── Resumen comparativo ──────────────────────────────────────
    comparar_series(resultados)

    print("\n" + "=" * 60)
    print(f"Análisis completado. Figuras en: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
