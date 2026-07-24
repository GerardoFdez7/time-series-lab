"""
src/__init__.py
Lab 1 – Series de Tiempo | CC3084 Data Science
Módulo principal del paquete src.
"""

from .data_loader import load_raw_data, clean_data, get_clean_data
from .series_builder import (
    build_total_series,
    build_via_series,
    build_tipo_viajero_series,
    build_all_series,
)
from .metrics import compute_mae, compute_rmse, compute_aic, compute_bic, model_summary

__all__ = [
    "load_raw_data",
    "clean_data",
    "get_clean_data",
    "build_total_series",
    "build_via_series",
    "build_tipo_viajero_series",
    "build_all_series",
    "compute_mae",
    "compute_rmse",
    "compute_aic",
    "compute_bic",
    "model_summary",
]
