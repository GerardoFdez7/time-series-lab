"""
src/metrics.py
Lab 1 – Series de Tiempo | CC3084 Data Science

Responsabilidad única: cálculo de métricas de evaluación de modelos.
Incluye MAE, RMSE, AIC y BIC.
"""

import numpy as np
import pandas as pd
from typing import Any


# ------------------------------------------------------------------ #
# Métricas de error de predicción                                      #
# ------------------------------------------------------------------ #


def compute_mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """
    Mean Absolute Error (MAE).

    MAE = (1/n) * Σ |y_true - y_pred|

    Parameters
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.

    Returns
    -------
    float
        MAE redondeado a 4 decimales.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.round(np.mean(np.abs(y_true - y_pred)), 4))


def compute_rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """
    Root Mean Square Error (RMSE).

    RMSE = sqrt((1/n) * Σ (y_true - y_pred)²)

    Parameters
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.

    Returns
    -------
    float
        RMSE redondeado a 4 decimales.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.round(np.sqrt(np.mean((y_true - y_pred) ** 2)), 4))


def compute_mape(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    MAPE = (100/n) * Σ |y_true - y_pred| / (|y_true| + ε)

    Parameters
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.
    epsilon : float
        Pequeño valor para evitar división por cero.

    Returns
    -------
    float
        MAPE en porcentaje, redondeado a 4 decimales.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(
        np.round(
            100 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon)), 4
        )
    )


# ------------------------------------------------------------------ #
# Métricas de información (criterios de selección de modelos)          #
# ------------------------------------------------------------------ #


def compute_aic(log_likelihood: float, n_params: int) -> float:
    """
    Akaike Information Criterion (AIC).

    AIC = -2 * log_likelihood + 2 * k

    Parameters
    ----------
    log_likelihood : float
        Log-verosimilitud del modelo.
    n_params : int
        Número de parámetros estimados (k).

    Returns
    -------
    float
        AIC del modelo (menor es mejor).
    """
    return float(np.round(-2 * log_likelihood + 2 * n_params, 4))


def compute_bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """
    Bayesian Information Criterion (BIC).

    BIC = -2 * log_likelihood + k * log(n)

    Parameters
    ----------
    log_likelihood : float
        Log-verosimilitud del modelo.
    n_params : int
        Número de parámetros estimados (k).
    n_obs : int
        Número de observaciones (n).

    Returns
    -------
    float
        BIC del modelo (menor es mejor).
    """
    return float(
        np.round(-2 * log_likelihood + n_params * np.log(n_obs), 4)
    )


# ------------------------------------------------------------------ #
# Resumen comparativo de modelos                                        #
# ------------------------------------------------------------------ #


def model_summary(
    model_name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    aic: float | None = None,
    bic: float | None = None,
) -> dict[str, Any]:
    """
    Calcula y retorna un diccionario con todas las métricas de un modelo.

    Parameters
    ----------
    model_name : str
        Nombre identificador del modelo (e.g., 'ARIMA(1,1,1)').
    y_true : array-like
        Valores reales del conjunto de prueba.
    y_pred : array-like
        Valores predichos por el modelo.
    aic : float | None
        AIC del modelo (si está disponible).
    bic : float | None
        BIC del modelo (si está disponible).

    Returns
    -------
    dict
        {'modelo': ..., 'MAE': ..., 'RMSE': ..., 'MAPE': ..., 'AIC': ..., 'BIC': ...}
    """
    return {
        "modelo": model_name,
        "MAE": compute_mae(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAPE (%)": compute_mape(y_true, y_pred),
        "AIC": round(aic, 4) if aic is not None else None,
        "BIC": round(bic, 4) if bic is not None else None,
    }


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """
    Imprime una tabla comparativa de modelos.

    Parameters
    ----------
    results : list[dict]
        Lista de diccionarios retornados por `model_summary`.
    """
    df = pd.DataFrame(results).set_index("modelo")
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)
    best_mae = df["MAE"].idxmin()
    best_rmse = df["RMSE"].idxmin()
    print(f"\n✓ Mejor MAE  : {best_mae}")
    print(f"✓ Mejor RMSE : {best_rmse}")
    if df["AIC"].notna().any():
        best_aic = df["AIC"].idxmin()
        print(f"✓ Mejor AIC  : {best_aic}")
    if df["BIC"].notna().any():
        best_bic = df["BIC"].idxmin()
        print(f"✓ Mejor BIC  : {best_bic}")
