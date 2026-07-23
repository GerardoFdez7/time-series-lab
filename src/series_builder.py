"""
src/series_builder.py
Lab 1 – Series de Tiempo | CC3084 Data Science

Responsabilidad única: construcción de series de tiempo mensuales
a partir del dataset limpio.
"""

import pandas as pd
from typing import NamedTuple


# ------------------------------------------------------------------ #
# Tipos de datos                                                        #
# ------------------------------------------------------------------ #


class SerieInfo(NamedTuple):
    """Metadata de una serie de tiempo."""

    nombre: str
    serie: pd.Series          # Índice DatetimeIndex frecuencia mensual ('MS')
    inicio: str               # Fecha de inicio (YYYY-MM)
    fin: str                  # Fecha de fin (YYYY-MM)
    frecuencia: str           # Siempre 'Mensual'
    categoria: str            # e.g. 'Total', 'Via', 'TipoViajero'
    descripcion: str          # Descripción larga para el informe


# ------------------------------------------------------------------ #
# Constante: tipos de viajero consistentes en todo el período          #
# ------------------------------------------------------------------ #

# Entre 2022-2023 "Viajero" se redefinió, excluyendo comercio fronterizo.
# Para comparabilidad histórica completa se recomienda Turista + Excursionista.
TIPOS_CONSISTENTES = {"turista", "excursionista"}


# ------------------------------------------------------------------ #
# Utilidades internas                                                   #
# ------------------------------------------------------------------ #


def _agg_mensual(df: pd.DataFrame, mask: pd.Series | None = None) -> pd.Series:
    """
    Agrega 'viajeros' por mes (columna 'fecha') aplicando una máscara opcional.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio (requiere columnas 'fecha' y 'viajeros').
    mask : pd.Series | None
        Máscara booleana de filas a incluir. Si None usa todo el df.

    Returns
    -------
    pd.Series
        Serie mensual con DatetimeIndex de frecuencia 'MS'.
    """
    subset = df[mask] if mask is not None else df
    serie = (
        subset.groupby("fecha")["viajeros"]
        .sum()
        .asfreq("MS")  # Frecuencia mensual, inicio del mes
        .fillna(0)      # Meses sin registros → 0
    )
    return serie


def _make_info(nombre: str, serie: pd.Series, categoria: str, descripcion: str) -> SerieInfo:
    """Construye un SerieInfo con los metadatos calculados automáticamente."""
    inicio = serie.index.min().strftime("%Y-%m")
    fin = serie.index.max().strftime("%Y-%m")
    return SerieInfo(
        nombre=nombre,
        serie=serie,
        inicio=inicio,
        fin=fin,
        frecuencia="Mensual",
        categoria=categoria,
        descripcion=descripcion,
    )


# ------------------------------------------------------------------ #
# Funciones públicas                                                   #
# ------------------------------------------------------------------ #


def build_total_series(df: pd.DataFrame) -> SerieInfo:
    """
    Serie obligatoria: total mensual de viajeros (Turista + Excursionista).

    Se usa Turista + Excursionista para garantizar comparabilidad en todo
    el período 2009-2026, dado el cambio de definición 2022-2023.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio.

    Returns
    -------
    SerieInfo
        Metadata + serie mensual total.
    """
    if "tipo_viajero" in df.columns:
        mask = df["tipo_viajero"].str.lower().isin(TIPOS_CONSISTENTES)
        serie = _agg_mensual(df, mask)
        descripcion = (
            "Total mensual de Turistas + Excursionistas. "
            "Se excluyen las categorías 'Viajero' y otras no consistentes "
            "en el período 2022-2023."
        )
    else:
        # Si no existe la columna, usar todo
        serie = _agg_mensual(df)
        descripcion = "Total mensual de todos los viajeros."

    return _make_info("Total Viajeros", serie, "Total", descripcion)


def build_via_series(df: pd.DataFrame) -> dict[str, SerieInfo]:
    """
    Series mensuales por vía de ingreso (Aérea, Terrestre, Marítima).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio (requiere columna 'via').

    Returns
    -------
    dict[str, SerieInfo]
        Diccionario { nombre_via: SerieInfo }.
    """
    if "via" not in df.columns:
        raise KeyError("Columna 'via' no encontrada. Revisar data_loader.COLUMN_ALIASES.")

    vias = df["via"].dropna().unique()
    result = {}

    for via in sorted(vias):
        mask = df["via"] == via
        serie = _agg_mensual(df, mask)
        info = _make_info(
            nombre=f"Vía {via}",
            serie=serie,
            categoria="Via",
            descripcion=f"Total mensual de viajeros que ingresan por vía {via}.",
        )
        result[via] = info
        print(f"[series_builder] Serie creada: Vía {via} | {info.inicio} → {info.fin}")

    return result


def build_tipo_viajero_series(df: pd.DataFrame) -> dict[str, SerieInfo]:
    """
    Series mensuales por tipo de viajero (Turista, Excursionista, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio (requiere columna 'tipo_viajero').

    Returns
    -------
    dict[str, SerieInfo]
        Diccionario { tipo_viajero: SerieInfo }.
    """
    if "tipo_viajero" not in df.columns:
        raise KeyError(
            "Columna 'tipo_viajero' no encontrada. Revisar data_loader.COLUMN_ALIASES."
        )

    tipos = df["tipo_viajero"].dropna().unique()
    result = {}

    for tipo in sorted(tipos):
        mask = df["tipo_viajero"] == tipo
        serie = _agg_mensual(df, mask)
        consistente = tipo.lower() in TIPOS_CONSISTENTES
        nota = (
            " NOTA: Consistente en todo el período."
            if consistente
            else " NOTA: Definición cambió entre 2022-2023, interpretar con cuidado."
        )
        info = _make_info(
            nombre=f"Tipo: {tipo}",
            serie=serie,
            categoria="TipoViajero",
            descripcion=f"Total mensual de viajeros de tipo '{tipo}'.{nota}",
        )
        result[tipo] = info
        print(f"[series_builder] Serie creada: Tipo {tipo} | {info.inicio} → {info.fin}")

    return result


def build_all_series(df: pd.DataFrame) -> dict[str, SerieInfo]:
    """
    Construye todas las series requeridas para el laboratorio:
    - Serie total (Turista + Excursionista)
    - Series por vía de ingreso
    - Series por tipo de viajero

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio.

    Returns
    -------
    dict[str, SerieInfo]
        Diccionario { nombre_serie: SerieInfo } con todas las series.
    """
    all_series: dict[str, SerieInfo] = {}

    print("\n[series_builder] ── Construyendo serie total ──")
    total = build_total_series(df)
    all_series[total.nombre] = total

    print("\n[series_builder] ── Construyendo series por vía ──")
    via_series = build_via_series(df)
    for info in via_series.values():
        all_series[info.nombre] = info

    print("\n[series_builder] ── Construyendo series por tipo de viajero ──")
    tipo_series = build_tipo_viajero_series(df)
    for info in tipo_series.values():
        all_series[info.nombre] = info

    print(f"\n[series_builder] Total de series construidas: {len(all_series)}")
    return all_series


def print_series_summary(all_series: dict[str, SerieInfo]) -> None:
    """Imprime una tabla resumen de todas las series."""
    print("\n" + "=" * 70)
    print(f"{'SERIE':<30} {'INICIO':<10} {'FIN':<10} {'FREC.':<10} {'CATEGORÍA'}")
    print("=" * 70)
    for info in all_series.values():
        print(
            f"{info.nombre:<30} {info.inicio:<10} {info.fin:<10} "
            f"{info.frecuencia:<10} {info.categoria}"
        )
    print("=" * 70)


# ------------------------------------------------------------------ #
# Ejecución directa (diagnóstico)                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_loader import get_clean_data

    df = get_clean_data()
    series = build_all_series(df)
    print_series_summary(series)
