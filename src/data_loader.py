"""
src/data_loader.py
Lab 1 – Series de Tiempo | CC3084 Data Science

Responsabilidad única: carga y limpieza del dataset de migración.
No genera gráficos ni hace análisis estadístico.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------------ #
# Configuración de rutas                                               #
# ------------------------------------------------------------------ #

# Raíz del proyecto (un nivel arriba de src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ruta al dataset. Se busca primero en data/raw/, luego en la raíz.
_POSSIBLE_PATHS = [
    PROJECT_ROOT / "data" / "raw" / "Base_Migracion_2009-2026jun.xlsx",
    PROJECT_ROOT
    / "Laboratorio 1. Series de Tiempo 2026 - Base_Migracion_2009-2026jun.xlsx",
]

# ------------------------------------------------------------------ #
# Mapeo de nombres de columna (normalización)                          #
# ------------------------------------------------------------------ #

# Mapeo de posibles nombres originales → nombre normalizado interno.
# Ajustar si el Excel tiene variaciones tipográficas.
COLUMN_ALIASES: dict[str, str] = {
    # Año
    "año": "anio",
    "ano": "anio",
    "year": "anio",
    # Mes código
    "mes cod": "mes_cod",
    "mes_cod": "mes_cod",
    "mescod": "mes_cod",
    # Mes nombre
    "mes": "mes",
    # Vía
    "vía": "via",
    "via": "via",
    # Frontera
    "frontera": "frontera",
    # País
    "país": "pais",
    "pais": "pais",
    "country": "pais",
    # Región
    "región": "region",
    "region": "region",
    # Región dos
    "región dos": "region_dos",
    "region dos": "region_dos",
    "region_dos": "region_dos",
    # Regiones OMT
    "regiones omt": "region_omt",
    "región omt": "region_omt",
    "region_omt": "region_omt",
    # MCEO
    "mceo": "mceo",
    # Agrupación residencia
    "agrupación residencia": "agrupacion_residencia",
    "agrupacion residencia": "agrupacion_residencia",
    "agrupacion_residencia": "agrupacion_residencia",
    # Tipo viajero
    "tipo de viajero": "tipo_viajero",
    "tipo_viajero": "tipo_viajero",
    "tipo viajero": "tipo_viajero",
    # Cantidad de viajeros
    "viajero": "viajeros",
    "viajeros": "viajeros",
    "cantidad": "viajeros",
    "total": "viajeros",
}


# ------------------------------------------------------------------ #
# Funciones públicas                                                   #
# ------------------------------------------------------------------ #


def find_data_path() -> Path:
    """Localiza el archivo de datos; lanza FileNotFoundError si no existe."""
    for path in _POSSIBLE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No se encontró el dataset. Asegúrate de copiar el .xlsx a data/raw/ "
        f"o a la raíz del proyecto.\nRutas revisadas:\n"
        + "\n".join(f"  - {p}" for p in _POSSIBLE_PATHS)
    )


def load_raw_data(filepath: Path | str | None = None) -> pd.DataFrame:
    """
    Carga el dataset crudo desde el archivo Excel.

    Parameters
    ----------
    filepath : Path | str | None
        Ruta explícita al archivo Excel. Si es None, se busca automáticamente.

    Returns
    -------
    pd.DataFrame
        Dataset tal cual viene del Excel, sin modificaciones.
    """
    path = Path(filepath) if filepath else find_data_path()
    print(f"[data_loader] Cargando datos desde: {path}")
    df = pd.read_excel(path, engine="openpyxl")
    print(f"[data_loader] Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columna a minúsculas sin espacios extra."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        normalized = COLUMN_ALIASES.get(col, col)
        rename_map[col] = normalized
    df = df.rename(columns=rename_map)
    return df


def _build_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea columna 'fecha' (primer día del mes) a partir de 'anio' y 'mes_cod'.
    Si 'mes_cod' no existe, intenta inferirla de la columna 'mes'.
    """
    df = df.copy()

    # Verificar que existen las columnas necesarias
    if "anio" not in df.columns:
        raise KeyError("No se encontró columna de año. Revisar COLUMN_ALIASES.")

    if "mes_cod" not in df.columns and "mes" in df.columns:
        # Mapeo de nombres de mes en español → número
        mes_map = {
            "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
            "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
            "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        df["mes_cod"] = (
            df["mes"].astype(str).str.strip().str.lower().map(mes_map)
        )

    if "mes_cod" not in df.columns:
        raise KeyError("No se encontró columna de mes. Revisar COLUMN_ALIASES.")

    df["fecha"] = pd.to_datetime(
        {
            "year": df["anio"].astype(int),
            "month": df["mes_cod"].astype(int),
            "day": 1,
        }
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y preprocesa el dataset.

    Pasos:
    1. Normalización de nombres de columna.
    2. Construcción de columna 'fecha'.
    3. Conversión de tipos.
    4. Reporte de valores nulos y duplicados.
    5. Eliminación de duplicados exactos.
    6. Manejo de valores nulos en 'viajeros' (se eliminan filas con NaN).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset crudo (salida de load_raw_data).

    Returns
    -------
    pd.DataFrame
        Dataset limpio listo para análisis.
    """
    df = _normalize_columns(df)
    df = _build_date_column(df)

    # ── Conversión de tipos ──────────────────────────────────────────
    if "viajeros" in df.columns:
        df["viajeros"] = pd.to_numeric(df["viajeros"], errors="coerce")

    # ── Reporte de calidad ───────────────────────────────────────────
    n_total = len(df)
    n_dup = df.duplicated().sum()
    n_null_viajeros = df["viajeros"].isna().sum() if "viajeros" in df.columns else 0

    print(f"[data_loader] Filas totales       : {n_total:,}")
    print(f"[data_loader] Duplicados exactos  : {n_dup:,}")
    print(f"[data_loader] Nulos en 'viajeros' : {n_null_viajeros:,}")

    # ── Limpieza ─────────────────────────────────────────────────────
    # Eliminar duplicados exactos
    if n_dup > 0:
        df = df.drop_duplicates()
        print(f"[data_loader] Duplicados eliminados: {n_dup:,}")

    # Eliminar filas sin cantidad de viajeros (no aportan información)
    if n_null_viajeros > 0 and "viajeros" in df.columns:
        df = df.dropna(subset=["viajeros"])
        print(f"[data_loader] Filas eliminadas por NaN en viajeros: {n_null_viajeros:,}")

    # ── Valores de viajeros negativos ────────────────────────────────
    if "viajeros" in df.columns:
        n_neg = (df["viajeros"] < 0).sum()
        if n_neg > 0:
            print(f"[data_loader] ADVERTENCIA: {n_neg:,} filas con viajeros < 0")
            df = df[df["viajeros"] >= 0]

    # ── Normalizar strings clave ─────────────────────────────────────
    str_cols = ["via", "tipo_viajero", "frontera", "pais", "region", "region_dos"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    print(f"[data_loader] Dataset limpio: {len(df):,} filas × {df.shape[1]} columnas")
    print(f"[data_loader] Período: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    return df


def get_clean_data(filepath: Path | str | None = None) -> pd.DataFrame:
    """
    Carga y limpia el dataset en un solo paso.

    Parameters
    ----------
    filepath : Path | str | None
        Ruta explícita al Excel. Si None, se busca automáticamente.

    Returns
    -------
    pd.DataFrame
        Dataset limpio listo para análisis.
    """
    raw = load_raw_data(filepath)
    return clean_data(raw)


# ------------------------------------------------------------------ #
# Ejecución directa (diagnóstico)                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    df = get_clean_data()
    print("\n=== Columnas disponibles ===")
    print(df.columns.tolist())
    print("\n=== Primeras filas ===")
    print(df.head())
    print("\n=== Tipos de viajero únicos ===")
    if "tipo_viajero" in df.columns:
        print(df["tipo_viajero"].value_counts())
    print("\n=== Vías de ingreso ===")
    if "via" in df.columns:
        print(df["via"].value_counts())
