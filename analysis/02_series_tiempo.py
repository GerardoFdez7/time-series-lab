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

    # ── CONSTRUCCIÓN DEL DASHBOARD CONSOLIDADO ───────────────────
    import matplotlib.gridspec as gridspec

    # Crear la figura maestra para esta serie (1 Dashboard = 1 Imagen)
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.2)

    # 1. Gráfico de la serie (Ocupa las dos columnas de la fila 0)
    ax_serie = fig.add_subplot(gs[0, :])
    ax_serie.plot(train.index, train.values / 1e3, label="Entrenamiento", color=color or PALETTE[0], linewidth=1.5)
    ax_serie.plot(test.index, test.values / 1e3, label="Prueba", color="tomato", linewidth=1.5)
    ax_serie.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-01"), alpha=0.15, color="red", label="COVID-19")
    ax_serie.set_title(f"1. Serie de Tiempo Original: {nombre}", fontweight="bold")
    ax_serie.set_ylabel("Miles de viajeros")
    ax_serie.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}K"))
    ax_serie.legend(fontsize=9)

    # 2. Descomposición: Tendencia y Estacionalidad (Fila 1)
    ax_trend = fig.add_subplot(gs[1, 0])
    ax_season = fig.add_subplot(gs[1, 1])

    if len(train) >= 24:
        try:
            decomp = seasonal_decompose(train, model="additive", period=12)
            ax_trend.plot(train.index, decomp.trend / 1e3, color=PALETTE[1])
            ax_trend.set_title("2a. Componente de Tendencia", fontweight="bold")
            ax_trend.set_ylabel("Miles de viajeros")
            
            ax_season.plot(train.index, decomp.seasonal / 1e3, color=PALETTE[2])
            ax_season.set_title("2b. Componente Estacional", fontweight="bold")
            
            # Magnitud para consola
            seasonal_strength = decomp.seasonal.std()
            trend_strength = decomp.trend.dropna().std()
            print(f"\n  Fuerza estacional (std): {seasonal_strength:,.0f}")
            print(f"  Fuerza de tendencia (std): {trend_strength:,.0f}")
        except Exception as e:
            print(f"  ADVERTENCIA descomposición: {e}")
            ax_trend.set_title("Descomposición no disponible")
            ax_season.set_title("Descomposición no disponible")
    else:
        print("  Descomposición omitida (menos de 24 observaciones en train).")

    # 3. Estacionariedad en Varianza (Rolling Stats y Log) (Fila 2)
    print("\n  ── Estacionariedad en varianza ──")
    rolling_std = train.rolling(window=12).std()
    rolling_mean = train.rolling(window=12).mean()
    cv_series = (rolling_std / rolling_mean).dropna()
    cv_overall = train.std() / train.mean()
    print(f"  Coef. Variación global: {cv_overall:.4f}")
    print(f"  CV rodante (media)    : {cv_series.mean():.4f}")
    
    needs_log = cv_series.std() > 0.15
    if needs_log:
        print("  → CONCLUSIÓN: Varianza NO constante. Se aplica transformación log.")
        train_log = np.log1p(train)
    else:
        train_log = train
        print("  → Varianza relativamente estable. No se requiere transformación.")

    ax_var = fig.add_subplot(gs[2, 0])
    ax_var.plot(rolling_mean.index, rolling_mean.values / 1e3, color="red", label="Media móvil")
    ax_var.plot(rolling_std.index, rolling_std.values / 1e3, color="steelblue", label="Std móvil")
    ax_var.set_title("3a. Varianza: Media y Desviación Estándar (12m)", fontweight="bold")
    ax_var.legend(fontsize=9)

    ax_log = fig.add_subplot(gs[2, 1])
    ax_log.plot(train_log.index, train_log.values, color=PALETTE[4], linewidth=1.2)
    ax_log.set_title("3b. Serie Transformada (Log)" if needs_log else "3b. Serie Original (CV aceptable)", fontweight="bold")

    # 4. Prueba ADF y Correlogramas ACF/PACF (Fila 3)
    print("\n  ── Prueba ADF (Dickey-Fuller Aumentada) ──")
    adf_result = adfuller(train.dropna(), autolag="AIC")
    adf_stat, p_value, n_lags = adf_result[0], adf_result[1], adf_result[2]
    
    print(f"  Estadístico ADF : {adf_stat:.6f} | p-value: {p_value:.6f}")
    is_stationary = p_value < 0.05
    if is_stationary:
        print("  → CONCLUSIÓN: La serie ES estacionaria (p < 0.05). d = 0")
        d_suggested = 0
    else:
        print("  → CONCLUSIÓN: La serie NO ES estacionaria (p ≥ 0.05). Se requiere diferenciación.")
        train_diff = train.diff().dropna()
        adf_diff = adfuller(train_diff, autolag="AIC")
        print(f"  ADF Diferenciada (d=1): p-value = {adf_diff[1]:.6f}")
        d_suggested = 1 if adf_diff[1] < 0.05 else 2

    n_lags_plot = min(36, len(train) // 2 - 1)
    
    ax_acf = fig.add_subplot(gs[3, 0])
    plot_acf(train.dropna(), lags=n_lags_plot, ax=ax_acf, color=color or PALETTE[0])
    ax_acf.set_title(f"4a. ACF (Autocorrelación) | ADF p-value: {p_value:.3f}", fontweight="bold")

    ax_pacf = fig.add_subplot(gs[3, 1])
    plot_pacf(train.dropna(), lags=n_lags_plot, ax=ax_pacf, method="ywm", color=color or PALETTE[0])
    ax_pacf.set_title(f"4b. PACF (Autocorrelación Parcial) | d sugerido: {d_suggested}", fontweight="bold")

    # Guardar el Dashboard completo
    fig.suptitle(f"Dashboard de Análisis de Serie: {nombre}", fontsize=18, fontweight="bold", y=0.98)
    # plt.tight_layout() no funciona tan bien con gridspec a veces, ajustamos márgenes manuales si hace falta
    fig.subplots_adjust(top=0.92)
    save_fig(f"dashboard_{safe_name}")

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

    # (Opcional) ACF sobre serie diferenciada si el p-value original fue alto, 
    # pero ya no generamos la imagen suelta para mantener el directorio limpio.

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
        "train_log": train_log,   # Serie con transformación aplicada (o igual a train si no aplica)
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

    # ── Sección 5: Análisis comparativo con evidencia estadística ──
    print("\n" + "=" * 60)
    print("ANÁLISIS COMPARATIVO (Sección 5 del lab)")
    print("=" * 60)

    # 5a-iii) Mayor volatilidad: Coeficiente de Variación global
    max_cv = max(resultados, key=lambda r: r["cv_global"])
    min_cv = min(resultados, key=lambda r: r["cv_global"])
    print(f"\n  Mayor volatilidad (CV): {max_cv['nombre']} → CV = {max_cv['cv_global']:.4f}")
    print(f"  Menor volatilidad (CV): {min_cv['nombre']} → CV = {min_cv['cv_global']:.4f}")

    # 5a-i) Mayor estacionalidad: fuerza estacional de la descomposición
    # Fuerza estacional = Var(estacional) / (Var(estacional) + Var(residuo))
    # Requiere descomposición — se calcula aquí desde los datos de train
    from statsmodels.tsa.seasonal import seasonal_decompose
    print("\n  ── Mayor estacionalidad (fuerza estacional) ──")
    fuerzas = []
    for r in resultados:
        try:
            tr = r["train"]
            if len(tr) >= 24:
                d = seasonal_decompose(tr, model="additive", period=12)
                var_s = d.seasonal.var()
                var_r = d.resid.dropna().var()
                fuerza = var_s / (var_s + var_r) if (var_s + var_r) > 0 else 0
            else:
                fuerza = 0.0
        except Exception:
            fuerza = 0.0
        fuerzas.append({"nombre": r["nombre"], "fuerza_estacional": round(fuerza, 4)})
    fuerzas_df = pd.DataFrame(fuerzas).sort_values("fuerza_estacional", ascending=False)
    print(fuerzas_df.to_string(index=False))
    print(f"  → Mayor estacionalidad: {fuerzas_df.iloc[0]['nombre']} (F={fuerzas_df.iloc[0]['fuerza_estacional']:.4f})")

    # 5a-ii) Mayor tendencia de crecimiento: pendiente de regresión lineal sobre la tendencia
    print("\n  ── Mayor tendencia de crecimiento (pendiente lineal) ──")
    pendientes = []
    for r in resultados:
        try:
            tr = r["train"]
            if len(tr) >= 24:
                d = seasonal_decompose(tr, model="additive", period=12)
                trend_vals = d.trend.dropna()
                x = np.arange(len(trend_vals))
                slope = np.polyfit(x, trend_vals.values, 1)[0]   # viajeros/mes
            else:
                slope = 0.0
        except Exception:
            slope = 0.0
        pendientes.append({"nombre": r["nombre"], "pendiente_mensual": round(slope, 1)})
    pend_df = pd.DataFrame(pendientes).sort_values("pendiente_mensual", ascending=False)
    print(pend_df.to_string(index=False))
    print(f"  → Mayor crecimiento: {pend_df.iloc[0]['nombre']} ({pend_df.iloc[0]['pendiente_mensual']:+,.0f} viajeros/mes)")

    # 5a-iv) Más afectada por pandemia: % de caída en 2020 vs 2019
    print("\n  ── Impacto de pandemia por serie (caída 2020 vs 2019) ──")
    impactos = []
    for r in resultados:
        tr = r["train"]
        val_2019 = tr[tr.index.year == 2019].sum()
        val_2020 = tr[tr.index.year == 2020].sum()
        pct_caida = ((val_2020 - val_2019) / val_2019 * 100) if val_2019 > 0 else 0
        impactos.append({"nombre": r["nombre"],
                         "viajeros_2019": int(val_2019),
                         "viajeros_2020": int(val_2020),
                         "caida_pct": round(pct_caida, 1)})
    impact_df = pd.DataFrame(impactos).sort_values("caida_pct")
    print(impact_df.to_string(index=False))
    print(f"  → Más afectada: {impact_df.iloc[0]['nombre']} ({impact_df.iloc[0]['caida_pct']:.1f}%)")

    # Gráfico comparativo de todas las métricas
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].barh(fuerzas_df["nombre"], fuerzas_df["fuerza_estacional"],
                 color=PALETTE[:len(fuerzas_df)], edgecolor="white")
    axes[0].set_title("Fuerza Estacional", fontweight="bold")
    axes[0].set_xlabel("Var(S) / (Var(S) + Var(R))")

    axes[1].barh(pend_df["nombre"], pend_df["pendiente_mensual"],
                 color=[PALETTE[0] if v >= 0 else PALETTE[3] for v in pend_df["pendiente_mensual"]],
                 edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Pendiente de Tendencia (viajeros/mes)", fontweight="bold")

    axes[2].barh(impact_df["nombre"], impact_df["caida_pct"],
                 color=PALETTE[:len(impact_df)], edgecolor="white")
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_title("Caída por Pandemia 2020 vs 2019 (%)", fontweight="bold")

    fig.suptitle("Análisis Comparativo entre Series", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("comparativo_metricas_seccion5")


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
