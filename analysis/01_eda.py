"""
analysis/01_eda.py
Lab 1 – Series de Tiempo | CC3084 Data Science
Persona 1: Análisis Exploratorio de Datos (EDA)

Ejecutar desde la raíz del proyecto:
    python analysis/01_eda.py

Los gráficos se guardan en reports/figures/eda_*.png
"""

import sys
from pathlib import Path

# Asegurar que src/ esté en el path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from src.data_loader import get_clean_data

# ------------------------------------------------------------------ #
# Configuración de estilos y rutas                                      #
# ------------------------------------------------------------------ #

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Paleta y estilo consistentes en todos los gráficos
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

PALETTE = sns.color_palette("tab10")


def save_fig(name: str) -> None:
    path = FIGURES_DIR / f"eda_{name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=120)
    print(f"  → Guardado: {path.name}")
    plt.close()


# ================================================================== #
# 1. ESTADÍSTICAS DESCRIPTIVAS                                         #
# ================================================================== #


def seccion_estadisticas_descriptivas(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("1. ESTADÍSTICAS DESCRIPTIVAS")
    print("=" * 60)

    print(f"\nShape del dataset: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"Período cubierto : {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"Años en el dataset: {df['anio'].nunique()}")

    print("\n── Estadísticas de 'viajeros' ──")
    desc = df["viajeros"].describe()
    print(desc.to_string())

    print(f"\nTotal de viajeros en el período: {df['viajeros'].sum():,.0f}")

    # Tabla de valores únicos por columna categórica
    cat_cols = [c for c in ["via", "tipo_viajero", "frontera", "region", "region_dos"] if c in df.columns]
    print("\n── Cardinalidad de columnas categóricas ──")
    for col in cat_cols:
        print(f"  {col:<25}: {df[col].nunique()} valores únicos")


# ================================================================== #
# 2. VALORES FALTANTES, DUPLICADOS Y ATÍPICOS                          #
# ================================================================== #


def seccion_calidad_datos(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("2. CALIDAD DE DATOS")
    print("=" * 60)

    # ── Valores faltantes ──────────────────────────────────────────
    print("\n── Valores faltantes por columna ──")
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    null_df = pd.DataFrame({"Nulos": null_counts, "% del total": null_pct})
    null_df = null_df[null_df["Nulos"] > 0]
    if null_df.empty:
        print("  No hay valores faltantes en el dataset limpio.")
    else:
        print(null_df.to_string())

    # ── Duplicados ─────────────────────────────────────────────────
    n_dup = df.duplicated().sum()
    print(f"\n── Duplicados exactos: {n_dup:,} ──")

    # ── Outliers en viajeros (IQR) ─────────────────────────────────
    Q1 = df["viajeros"].quantile(0.25)
    Q3 = df["viajeros"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df["viajeros"] < lower) | (df["viajeros"] > upper)]
    print(f"\n── Outliers en 'viajeros' (IQR) ──")
    print(f"  Límite inferior: {lower:,.0f}")
    print(f"  Límite superior: {upper:,.0f}")
    print(f"  Filas con outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")

    # Gráfico: boxplot de viajeros por año
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    df.boxplot(column="viajeros", ax=ax)
    ax.set_title("Distribución de viajeros por fila")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("")

    ax = axes[1]
    if "anio" in df.columns:
        yearly_total = df.groupby("anio")["viajeros"].sum() / 1e6
        yearly_total.plot(kind="bar", ax=ax, color=PALETTE[0], edgecolor="white")
        ax.set_title("Total anual de viajeros (millones)")
        ax.set_xlabel("Año")
        ax.set_ylabel("Millones de viajeros")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Calidad de Datos y Distribución por Año", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("01_calidad_datos")


# ================================================================== #
# 3. COMPORTAMIENTO TEMPORAL                                            #
# ================================================================== #


def seccion_comportamiento_temporal(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("3. COMPORTAMIENTO TEMPORAL")
    print("=" * 60)

    # Serie mensual total (Turista + Excursionista para consistencia)
    if "tipo_viajero" in df.columns:
        mask = df["tipo_viajero"].str.lower().isin({"turista", "excursionista"})
        serie_total = df[mask].groupby("fecha")["viajeros"].sum()
    else:
        serie_total = df.groupby("fecha")["viajeros"].sum()

    print(f"  Meses con datos: {len(serie_total)}")
    print(f"  Máximo mensual : {serie_total.max():,.0f} ({serie_total.idxmax().strftime('%Y-%m')})")
    print(f"  Mínimo mensual : {serie_total.min():,.0f} ({serie_total.idxmin().strftime('%Y-%m')})")

    # Gráfico 1: Serie temporal total
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(serie_total.index, serie_total.values / 1e3, color=PALETTE[0], linewidth=1.5, label="Turistas + Excursionistas")
    # Marcar pandemia
    ax.axvspan(
        pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-01"),
        alpha=0.15, color="red", label="COVID-19 (Mar 2020 – Dic 2021)"
    )
    ax.set_title("Comportamiento temporal: Total mensual de viajeros (2009–2026)", fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Miles de viajeros")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}K"))
    ax.legend()
    plt.tight_layout()
    save_fig("02_serie_total_mensual")

    # Gráfico 2: Estacionalidad anual (promedio por mes)
    if "mes_cod" in df.columns:
        if "tipo_viajero" in df.columns:
            df_filt = df[df["tipo_viajero"].str.lower().isin({"turista", "excursionista"})]
        else:
            df_filt = df

        monthly_avg = df_filt.groupby("mes_cod")["viajeros"].mean() / 1e3
        meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                 "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(
            range(1, len(monthly_avg) + 1),
            monthly_avg.values,
            color=PALETTE[:len(monthly_avg)],
            edgecolor="white",
        )
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(meses[:len(monthly_avg)])
        ax.set_title("Promedio mensual de viajeros (patrón estacional)", fontweight="bold")
        ax.set_ylabel("Miles de viajeros (promedio)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}K"))
        plt.tight_layout()
        save_fig("03_patron_estacional")

    # Gráfico 3: Heatmap año × mes
    if "anio" in df.columns and "mes_cod" in df.columns:
        if "tipo_viajero" in df.columns:
            df_filt = df[df["tipo_viajero"].str.lower().isin({"turista", "excursionista"})]
        else:
            df_filt = df

        pivot = df_filt.pivot_table(
            values="viajeros", index="anio", columns="mes_cod", aggfunc="sum"
        ) / 1e3
        pivot.columns = meses[:pivot.shape[1]]

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            linewidths=0.3,
            annot=True,
            fmt=".0f",
            annot_kws={"size": 7},
            cbar_kws={"label": "Miles de viajeros"},
        )
        ax.set_title("Heatmap: Viajeros por Año y Mes (miles)", fontweight="bold")
        ax.set_xlabel("Mes")
        ax.set_ylabel("Año")
        plt.tight_layout()
        save_fig("04_heatmap_anio_mes")

        print("\n  INTERPRETACIÓN:")
        print("  - Pico de viajeros típicamente en los meses de verano boreal (Jul-Ago)")
        print("    y temporada de fin de año (Dic).")
        print("  - Caída brusca en 2020 (Mar-Jun) por COVID-19.")
        print("  - Recuperación gradual 2021-2022; niveles pre-pandemia 2023+.")


# ================================================================== #
# 4. ANÁLISIS POR PAÍSES                                               #
# ================================================================== #


def seccion_paises(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("4. PAÍSES CON MAYOR CANTIDAD DE VIAJEROS")
    print("=" * 60)

    if "pais" not in df.columns:
        print("  Columna 'pais' no encontrada.")
        return

    # Filtrar valores que no son países reales (artefacto del dataset:
    # cruceristas tienen País='Cruceristas' en lugar de un país de origen)
    PAISES_EXCLUIR = {"cruceristas", "cruceros"}
    df_paises = df[~df["pais"].str.lower().isin(PAISES_EXCLUIR)]

    top_paises = (
        df_paises.groupby("pais")["viajeros"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )

    print("\n  NOTA: 'Cruceristas' excluido del ranking de países (artefacto del dataset)")
    print("  Los cruceristas tienen País='Cruceristas' en lugar de su país de origen.")
    print("\n── Top 15 países (excluye 'Cruceristas') ──")
    for i, (pais, total) in enumerate(top_paises.items(), 1):
        print(f"  {i:2}. {pais:<35} {total:>12,.0f}")

    # Gráfico: barras horizontales Top 15
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[0]] * 3 + [PALETTE[1]] * (len(top_paises) - 3)
    bars = ax.barh(
        top_paises.index[::-1],
        top_paises.values[::-1] / 1e6,
        color=colors[::-1],
        edgecolor="white",
    )
    ax.set_title("Top 15 Países por Total de Viajeros (2009–2026)\n(excluye 'Cruceristas' como entrada de País)", fontweight="bold")
    ax.set_xlabel("Millones de viajeros")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))

    # Etiquetas en las barras
    for bar, val in zip(bars, top_paises.values[::-1] / 1e6):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}M", va="center", fontsize=9)

    plt.tight_layout()
    save_fig("05_top_paises")


# ================================================================== #
# 5. ANÁLISIS POR REGIONES                                             #
# ================================================================== #


def seccion_regiones(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("5. REGIONES CON MAYOR CANTIDAD DE VIAJEROS")
    print("=" * 60)

    for col_region in ["region_dos", "region"]:
        if col_region in df.columns:
            # Filtrar valores de región inválidos ('0', NaN, 'nan', categorías no geográficas)
            REGIONES_EXCLUIR = {"0", "nan", "", "cruceristas", "cruceros"}
            df_reg = df[~df[col_region].astype(str).str.strip().str.lower().isin(REGIONES_EXCLUIR)]

            top_regiones = (
                df_reg.groupby(col_region)["viajeros"]
                .sum()
                .sort_values(ascending=False)
            )

            print(f"\n── Por '{col_region}' (filtrado valor '0' inválido) ──")
            for region, total in top_regiones.items():
                pct = total / top_regiones.sum() * 100
                print(f"  {region:<35} {total:>12,.0f}  ({pct:.1f}%)")

            # Gráfico: pie chart + barras
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax = axes[0]
            ax.pie(
                top_regiones.values,
                labels=top_regiones.index,
                autopct="%1.1f%%",
                colors=PALETTE[:len(top_regiones)],
                startangle=90,
            )
            ax.set_title(f"Distribución por {col_region}", fontweight="bold")

            ax = axes[1]
            bars = ax.barh(
                top_regiones.index[::-1],
                top_regiones.values[::-1] / 1e6,
                color=PALETTE[:len(top_regiones)][::-1],
                edgecolor="white",
            )
            ax.set_title(f"Total acumulado por {col_region}", fontweight="bold")
            ax.set_xlabel("Millones de viajeros")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))

            plt.suptitle(f"Análisis por Región: {col_region}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            save_fig(f"06_regiones_{col_region}")


# ================================================================== #
# 6. VÍAS DE INGRESO Y FRONTERAS                                       #
# ================================================================== #


def seccion_vias_fronteras(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("6. VÍAS DE INGRESO Y FRONTERAS")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Vías de ingreso ───────────────────────────────────────────
    if "via" in df.columns:
        via_counts = df.groupby("via")["viajeros"].sum().sort_values(ascending=False)
        print("\n── Vías de ingreso ──")
        for via, total in via_counts.items():
            pct = total / via_counts.sum() * 100
            print(f"  {via:<20} {total:>12,.0f}  ({pct:.1f}%)")

        ax = axes[0]
        ax.pie(
            via_counts.values,
            labels=via_counts.index,
            autopct="%1.1f%%",
            colors=PALETTE[:len(via_counts)],
            startangle=90,
            explode=[0.03] * len(via_counts),
        )
        ax.set_title("Distribución por Vía de Ingreso", fontweight="bold")

    # ── Fronteras ─────────────────────────────────────────────────
    if "frontera" in df.columns:
        top_fronteras = (
            df.groupby("frontera")["viajeros"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        print("\n── Top 10 fronteras ──")
        for frontera, total in top_fronteras.items():
            pct = total / df["viajeros"].sum() * 100
            print(f"  {frontera:<35} {total:>12,.0f}  ({pct:.1f}%)")

        ax = axes[1]
        bars = ax.barh(
            top_fronteras.index[::-1],
            top_fronteras.values[::-1] / 1e6,
            color=PALETTE[:len(top_fronteras)][::-1],
            edgecolor="white",
        )
        ax.set_title("Top 10 Fronteras por Total de Viajeros", fontweight="bold")
        ax.set_xlabel("Millones de viajeros")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))

    plt.suptitle("Vías de Ingreso y Fronteras (2009–2026)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("07_vias_fronteras")


# ================================================================== #
# 7. TIPO DE VIAJERO                                                    #
# ================================================================== #


def seccion_tipo_viajero(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("7. TIPO DE VIAJERO")
    print("=" * 60)

    if "tipo_viajero" not in df.columns:
        print("  Columna 'tipo_viajero' no encontrada.")
        return

    tipo_counts = df.groupby("tipo_viajero")["viajeros"].sum().sort_values(ascending=False)
    print("\n── Total por tipo de viajero ──")
    for tipo, total in tipo_counts.items():
        pct = total / tipo_counts.sum() * 100
        print(f"  {tipo:<25} {total:>12,.0f}  ({pct:.1f}%)")

    # Gráfico: evolución por tipo de viajero
    if "anio" in df.columns:
        tipo_anual = (
            df.groupby(["anio", "tipo_viajero"])["viajeros"]
            .sum()
            .unstack(fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(14, 5))
        for i, col in enumerate(tipo_anual.columns):
            ax.plot(tipo_anual.index, tipo_anual[col] / 1e6,
                    label=col, marker="o", markersize=4,
                    color=PALETTE[i % len(PALETTE)])

        # Marcar pandemia
        ax.axvspan(2020, 2021.9, alpha=0.15, color="red", label="COVID-19")
        # Marcar cambio de definición
        ax.axvline(x=2022.5, color="orange", linestyle="--", linewidth=1.5,
                   label="Cambio definición 'Viajero' (2022-2023)")

        ax.set_title("Evolución anual por Tipo de Viajero", fontweight="bold")
        ax.set_xlabel("Año")
        ax.set_ylabel("Millones de viajeros")
        ax.legend(loc="upper left", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
        plt.tight_layout()
        save_fig("08_tipo_viajero_evolucion")

        print("\n  INTERPRETACIÓN:")
        print("  - La categoría 'Viajero' cae bruscamente en 2023 por cambio de definición,")
        print("    NO por caída real del turismo.")
        print("  - Para comparaciones longitudinales usar Turista + Excursionista.")


# ================================================================== #
# 8. PANDEMIA: ANÁLISIS PRE/DURANTE/POST                               #
# ================================================================== #


def seccion_pandemia(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("8. IMPACTO DE LA PANDEMIA COVID-19")
    print("=" * 60)

    if "tipo_viajero" in df.columns:
        df_filt = df[df["tipo_viajero"].str.lower().isin({"turista", "excursionista"})]
    else:
        df_filt = df

    def label_periodo(anio):
        if anio < 2020:
            return "Pre-pandemia (2009-2019)"
        elif anio <= 2021:
            return "Pandemia (2020-2021)"
        else:
            return "Post-pandemia (2022-2026)"

    df_filt = df_filt.copy()
    df_filt["periodo"] = df_filt["anio"].apply(label_periodo)

    periodo_stats = df_filt.groupby("periodo")["viajeros"].agg(["sum", "mean", "std"])
    periodo_stats.columns = ["Total", "Promedio mensual", "Desv. estándar"]
    print(periodo_stats.to_string())

    # Comparación pre/post pandemia por mes (promedio)
    if "mes_cod" in df_filt.columns:
        pivot_periodo = df_filt.groupby(["periodo", "mes_cod"])["viajeros"].mean().unstack()
        meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                 "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

        fig, ax = plt.subplots(figsize=(12, 5))
        for i, (periodo, row) in enumerate(pivot_periodo.iterrows()):
            ax.plot(
                range(1, len(row) + 1), row.values / 1e3,
                label=periodo, marker="o", linewidth=2,
                color=PALETTE[i],
            )
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(meses)
        ax.set_title("Patrón estacional: Pre/Durante/Post pandemia", fontweight="bold")
        ax.set_ylabel("Miles de viajeros (promedio)")
        ax.legend()
        plt.tight_layout()
        save_fig("09_pandemia_comparacion")


# ================================================================== #
# MAIN                                                                  #
# ================================================================== #


def main() -> None:
    print("=" * 60)
    print("ANÁLISIS EXPLORATORIO DE DATOS – Viajeros Internacionales")
    print("Lab 1 | CC3084 Data Science | UVG 2026")
    print("=" * 60)

    # Carga de datos
    df = get_clean_data()

    # Ejecutar todas las secciones
    seccion_estadisticas_descriptivas(df)
    seccion_calidad_datos(df)
    seccion_comportamiento_temporal(df)
    seccion_paises(df)
    seccion_regiones(df)
    seccion_vias_fronteras(df)
    seccion_tipo_viajero(df)
    seccion_pandemia(df)

    print("\n" + "=" * 60)
    print(f"EDA completado. Figuras guardadas en: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
