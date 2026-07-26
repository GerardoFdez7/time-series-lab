"""
src/models.py
Lab 1 – Series de Tiempo | CC3084 Data Science

Responsabilidad única: wrappers de modelos de series de tiempo.
Cada función recibe train/test y retorna predicciones + métricas.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Any

# ── Statsmodels ──────────────────────────────────────────────────────
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ── pmdarima ─────────────────────────────────────────────────────────
try:
    from pmdarima import auto_arima as _auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    warnings.warn("pmdarima no instalado. auto_arima no disponible.", ImportWarning)

# ── Prophet ──────────────────────────────────────────────────────────
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("prophet no instalado. Prophet no disponible.", ImportWarning)

from .metrics import model_summary


# ------------------------------------------------------------------ #
# Utilidades internas                                                   #
# ------------------------------------------------------------------ #


def _align_predictions(pred: pd.Series, test: pd.Series) -> pd.Series:
    """Alinea las predicciones al índice del conjunto de prueba."""
    return pred.reindex(test.index)


# ------------------------------------------------------------------ #
# ARIMA / SARIMA                                                       #
# ------------------------------------------------------------------ #


def fit_arima(
    train: pd.Series,
    test: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Ajusta un modelo ARIMA/SARIMA y predice sobre el conjunto de prueba.

    Parameters
    ----------
    train : pd.Series
        Serie de entrenamiento con DatetimeIndex.
    test : pd.Series
        Serie de prueba con DatetimeIndex.
    order : tuple (p, d, q)
        Orden del modelo ARIMA.
    seasonal_order : tuple (P, D, Q, s)
        Orden estacional. (0,0,0,0) = sin estacionalidad.
    model_name : str | None
        Nombre del modelo. Si None, se construye automáticamente.

    Returns
    -------
    dict
        {'modelo': ..., 'fit': ..., 'predicciones': ..., 'metricas': ...}
    """
    p, d, q = order
    P, D, Q, s = seasonal_order

    if model_name is None:
        if s > 0:
            model_name = f"SARIMA({p},{d},{q})({P},{D},{Q})[{s}]"
        else:
            model_name = f"ARIMA({p},{d},{q})"

    print(f"[models] Ajustando {model_name}...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)

    n_steps = len(test)
    pred = fit.forecast(steps=n_steps)
    pred.index = test.index

    metricas = model_summary(
        model_name=model_name,
        y_true=test,
        y_pred=pred,
        aic=fit.aic,
        bic=fit.bic,
    )

    print(f"[models] {model_name} → MAE={metricas['MAE']:,.0f} | AIC={fit.aic:.2f}")

    return {
        "modelo": model_name,
        "fit": fit,
        "predicciones": pred,
        "metricas": metricas,
    }


def fit_auto_arima(
    train: pd.Series,
    test: pd.Series,
    seasonal: bool = True,
    m: int = 12,
) -> dict[str, Any]:
    """
    Usa auto_arima de pmdarima para encontrar el mejor modelo ARIMA/SARIMA.

    Parameters
    ----------
    train : pd.Series
        Serie de entrenamiento.
    test : pd.Series
        Serie de prueba.
    seasonal : bool
        Si True, busca componente estacional.
    m : int
        Período estacional (12 para mensual).

    Returns
    -------
    dict
        {'modelo': ..., 'fit': ..., 'predicciones': ..., 'metricas': ...}
    """
    if not PMDARIMA_AVAILABLE:
        raise ImportError("Instalar pmdarima: pip install pmdarima")

    print("[models] Ejecutando auto_arima (puede tardar)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _auto_arima(
            train,
            seasonal=seasonal,
            m=m,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aic",
        )

    order = fit.order
    seasonal_order = fit.seasonal_order
    model_name = f"AutoARIMA({order[0]},{order[1]},{order[2]})({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]})[{seasonal_order[3]}]"

    n_steps = len(test)
    pred_arr = fit.predict(n_periods=n_steps)
    pred = pd.Series(pred_arr, index=test.index)

    metricas = model_summary(
        model_name=model_name,
        y_true=test,
        y_pred=pred,
        aic=fit.aic(),
        bic=fit.bic(),
    )

    print(f"[models] {model_name} → MAE={metricas['MAE']:,.0f} | AIC={fit.aic():.2f}")
    print(f"[models] Parámetros auto_arima: orden={order}, estacional={seasonal_order}")

    return {
        "modelo": model_name,
        "fit": fit,
        "predicciones": pred,
        "metricas": metricas,
    }


# ------------------------------------------------------------------ #
# Holt-Winters (Suavizamiento Exponencial Triple)                      #
# ------------------------------------------------------------------ #


def fit_holt_winters(
    train: pd.Series,
    test: pd.Series,
    trend: str = "add",
    seasonal: str = "add",
    seasonal_periods: int = 12,
) -> dict[str, Any]:
    """
    Ajusta modelo Holt-Winters (suavizamiento exponencial triple).

    Parameters
    ----------
    train : pd.Series
        Serie de entrenamiento.
    test : pd.Series
        Serie de prueba.
    trend : str
        'add' (aditivo) o 'mul' (multiplicativo).
    seasonal : str
        'add' o 'mul'.
    seasonal_periods : int
        Número de períodos por ciclo (12 para mensual).

    Returns
    -------
    dict
        {'modelo': ..., 'fit': ..., 'predicciones': ..., 'metricas': ...}
    """
    model_name = f"HoltWinters(trend={trend}, seasonal={seasonal})"
    print(f"[models] Ajustando {model_name}...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
        )
        fit = model.fit(optimized=True)

    n_steps = len(test)
    pred = fit.forecast(n_steps)
    pred.index = test.index

    metricas = model_summary(
        model_name=model_name,
        y_true=test,
        y_pred=pred,
        aic=fit.aic,
        bic=fit.bic,
    )

    print(f"[models] {model_name} → MAE={metricas['MAE']:,.0f}")

    return {
        "modelo": model_name,
        "fit": fit,
        "predicciones": pred,
        "metricas": metricas,
    }


# ------------------------------------------------------------------ #
# Suavizamiento Exponencial Simple                                      #
# ------------------------------------------------------------------ #


def fit_simple_exp_smoothing(
    train: pd.Series,
    test: pd.Series,
) -> dict[str, Any]:
    """
    Ajusta suavizamiento exponencial simple (sin tendencia ni estacionalidad).

    Parameters
    ----------
    train : pd.Series
        Serie de entrenamiento.
    test : pd.Series
        Serie de prueba.

    Returns
    -------
    dict
        {'modelo': ..., 'fit': ..., 'predicciones': ..., 'metricas': ...}
    """
    model_name = "SuavizamientoExponencial"
    print(f"[models] Ajustando {model_name}...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(train, trend=None, seasonal=None)
        fit = model.fit(optimized=True)

    n_steps = len(test)
    pred = fit.forecast(n_steps)
    pred.index = test.index

    metricas = model_summary(
        model_name=model_name,
        y_true=test,
        y_pred=pred,
        aic=fit.aic,
        bic=fit.bic,
    )

    print(f"[models] {model_name} → MAE={metricas['MAE']:,.0f}")
    return {
        "modelo": model_name,
        "fit": fit,
        "predicciones": pred,
        "metricas": metricas,
    }


# ------------------------------------------------------------------ #
# Seasonal Naïve                                                        #
# ------------------------------------------------------------------ #


def fit_seasonal_naive(
    train: pd.Series,
    test: pd.Series,
    m: int = 12,
) -> dict[str, Any]:
    """
    Seasonal Naïve: predice usando el mismo valor del año anterior.
    Predicción para período t = valor de t - m.

    Parameters
    ----------
    train : pd.Series
        Serie de entrenamiento.
    test : pd.Series
        Serie de prueba.
    m : int
        Período estacional (12 = anual mensual).

    Returns
    -------
    dict
        {'modelo': ..., 'fit': None, 'predicciones': ..., 'metricas': ...}
    """
    model_name = "SeasonalNaive"
    print(f"[models] Calculando {model_name}...")

    # El pronóstico repite el último ciclo estacional completo observado en
    # entrenamiento de forma indefinida (estándar en fpp2/fpp3): para el paso
    # h del conjunto de prueba se usa train[len(train) - m + (h % m)].
    pred_values = []
    for i in range(len(test)):
        past_idx = len(train) - m + (i % m)
        pred_values.append(train.iloc[past_idx])

    pred = pd.Series(pred_values, index=test.index)

    metricas = model_summary(
        model_name=model_name,
        y_true=test,
        y_pred=pred,
    )

    print(f"[models] {model_name} → MAE={metricas['MAE']:,.0f}")
    return {
        "modelo": model_name,
        "fit": None,
        "predicciones": pred,
        "metricas": metricas,
    }


# ------------------------------------------------------------------ #
# Prophet                                                               #
# ------------------------------------------------------------------ #


def fit_prophet(
    train: pd.Series,
    test: pd.Series,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = False,
    daily_seasonality: bool = False,
) -> dict[str, Any]:
    """
    Ajusta modelo Prophet de Meta/Facebook.

    Parameters
    ----------
    train : pd.Series
        Serie de entrenamiento con DatetimeIndex mensual.
    test : pd.Series
        Serie de prueba con DatetimeIndex mensual.
    yearly_seasonality : bool
        Incluir estacionalidad anual.
    weekly_seasonality : bool
        Incluir estacionalidad semanal (normalmente False para datos mensuales).
    daily_seasonality : bool
        Incluir estacionalidad diaria (normalmente False para datos mensuales).

    Returns
    -------
    dict
        {'modelo': ..., 'fit': ..., 'predicciones': ..., 'metricas': ...}
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Instalar prophet: pip install prophet")

    model_name = "Prophet"
    print(f"[models] Ajustando {model_name}...")

    # Prophet requiere DataFrame con columnas 'ds' y 'y'
    train_df = pd.DataFrame({"ds": train.index, "y": train.values})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
        )
        model.fit(train_df)

    # Crear DataFrame futuro con las fechas del test
    future_df = pd.DataFrame({"ds": test.index})
    forecast = model.predict(future_df)
    pred = pd.Series(forecast["yhat"].values, index=test.index)

    metricas = model_summary(
        model_name=model_name,
        y_true=test,
        y_pred=pred,
    )

    print(f"[models] {model_name} → MAE={metricas['MAE']:,.0f}")
    return {
        "modelo": model_name,
        "fit": model,
        "predicciones": pred,
        "metricas": metricas,
        "forecast_df": forecast,
    }
