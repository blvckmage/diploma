#!/usr/bin/env python3
"""
ARIMA-прогнозирование и сравнение с наивными baseline-ами.
Используется как академическое сравнение с ML-подходом.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# ADF-тест на стационарность
# ---------------------------------------------------------------------------

def adf_test(series: pd.Series, name: str = '') -> Dict:
    """Augmented Dickey-Fuller тест. p < 0.05 → ряд стационарен."""
    from statsmodels.tsa.stattools import adfuller
    clean = series.dropna()
    if len(clean) < 8:
        return {'skill': name, 'adf_stat': np.nan, 'p_value': np.nan,
                'is_stationary': False, 'lags_used': 0}
    if clean.nunique() <= 1:
        return {'skill': name, 'adf_stat': np.nan, 'p_value': 1.0,
                'is_stationary': False, 'lags_used': 0,
                'critical_1pct': np.nan, 'critical_5pct': np.nan}
    result = adfuller(clean, autolag='AIC')
    return {
        'skill':          name,
        'adf_stat':       round(result[0], 4),
        'p_value':        round(result[1], 4),
        'is_stationary':  result[1] < 0.05,
        'lags_used':      result[2],
        'critical_1pct':  round(result[4]['1%'], 4),
        'critical_5pct':  round(result[4]['5%'], 4),
    }


def run_adf_for_all_skills(weekly_demand: pd.DataFrame) -> pd.DataFrame:
    """Запускает ADF-тест для каждого навыка (по vacancy_count)."""
    print("🔬 ADF-тест стационарности временных рядов...")
    rows = []
    for skill in weekly_demand['skill'].unique():
        s = (weekly_demand[weekly_demand['skill'] == skill]
             .sort_values(['year', 'week'])['vacancy_count'])
        rows.append(adf_test(s, skill))
    df = pd.DataFrame(rows).sort_values('p_value')
    n_stat = df['is_stationary'].sum()
    print(f"   Стационарных рядов: {n_stat}/{len(df)}")
    return df


# ---------------------------------------------------------------------------
# Наивные baseline-модели
# ---------------------------------------------------------------------------

def persistence_forecast(train: np.ndarray, n_steps: int) -> np.ndarray:
    """Persistence (naïve): предсказание = последнее известное значение."""
    return np.full(n_steps, train[-1])


def ma_forecast(train: np.ndarray, n_steps: int, window: int = 4) -> np.ndarray:
    """Moving Average (MA-k): предсказание = среднее за последние k шагов."""
    w = min(window, len(train))
    return np.full(n_steps, np.mean(train[-w:]))


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

def _best_arima_order(series: np.ndarray) -> Tuple[int, int, int]:
    """Подбирает (p,d,q) перебором по AIC среди простых вариантов."""
    from statsmodels.tsa.arima.model import ARIMA
    candidates = [(1,1,0),(1,1,1),(0,1,1),(2,1,0),(0,1,2),(1,0,0)]
    best_aic, best_order = np.inf, (1, 1, 1)
    for order in candidates:
        try:
            aic = ARIMA(series, order=order).fit().aic
            if aic < best_aic:
                best_aic, best_order = aic, order
        except Exception:
            pass
    return best_order


def fit_arima(train: np.ndarray, order: Optional[Tuple] = None):
    from statsmodels.tsa.arima.model import ARIMA
    if order is None:
        order = _best_arima_order(train)
    try:
        return ARIMA(train, order=order).fit(), order
    except Exception:
        return None, order


# ---------------------------------------------------------------------------
# Сравнение моделей для одного навыка
# ---------------------------------------------------------------------------

def compare_one_skill(
    weekly_demand: pd.DataFrame,
    skill: str,
    n_test: int = 4,
    gb_model=None,
    feature_cols: List[str] = None,
) -> Optional[Dict]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    sk = (weekly_demand[weekly_demand['skill'] == skill]
          .sort_values(['year', 'week']))
    if len(sk) < n_test + 6:
        return None

    y = sk['demand_index'].values
    train_y, test_y = y[:-n_test], y[-n_test:]

    row = {'skill': skill, 'n_train': len(train_y), 'n_test': n_test}

    # Persistence
    p = persistence_forecast(train_y, n_test)
    row['mae_persistence'] = round(mean_absolute_error(test_y, p), 5)
    row['rmse_persistence'] = round(np.sqrt(mean_squared_error(test_y, p)), 5)

    # MA-4
    m = ma_forecast(train_y, n_test, window=4)
    row['mae_ma4'] = round(mean_absolute_error(test_y, m), 5)
    row['rmse_ma4'] = round(np.sqrt(mean_squared_error(test_y, m)), 5)

    # ARIMA
    fitted, order = fit_arima(train_y)
    row['arima_order'] = str(order)
    if fitted is not None:
        try:
            pred_a = fitted.forecast(steps=n_test)
            row['mae_arima']  = round(mean_absolute_error(test_y, pred_a), 5)
            row['rmse_arima'] = round(np.sqrt(mean_squared_error(test_y, pred_a)), 5)
            row['arima_aic']  = round(fitted.aic, 2)
        except Exception:
            row['mae_arima'] = row['rmse_arima'] = np.nan
    else:
        row['mae_arima'] = row['rmse_arima'] = np.nan

    # GB-модель (если передана)
    if gb_model is not None and feature_cols is not None:
        test_X = sk[feature_cols].values[-n_test:]
        try:
            pred_gb = gb_model.predict(test_X)
            row['mae_gb']  = round(mean_absolute_error(test_y, pred_gb), 5)
            row['rmse_gb'] = round(np.sqrt(mean_squared_error(test_y, pred_gb)), 5)
        except Exception:
            row['mae_gb'] = row['rmse_gb'] = np.nan

    return row


def run_arima_comparison(
    weekly_demand: pd.DataFrame,
    top_skills: List[str],
    n_test: int = 4,
    gb_model=None,
    feature_cols: List[str] = None,
    output_path: str = 'reports/arima_comparison.csv',
) -> pd.DataFrame:
    print("📊 Сравнение ARIMA vs Baselines vs ML...")
    rows = []
    for skill in top_skills:
        r = compare_one_skill(weekly_demand, skill, n_test, gb_model, feature_cols)
        if r:
            rows.append(r)
            mae_a = r.get('mae_arima', float('nan'))
            print(f"   {skill:<18} Persist={r['mae_persistence']:.4f}  "
                  f"MA4={r['mae_ma4']:.4f}  "
                  f"ARIMA={mae_a:.4f}  "
                  f"GB={r.get('mae_gb', float('nan')):.4f}")
    df = pd.DataFrame(rows)
    import os; os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"   ✅ Сохранено: {output_path}")
    return df


# ---------------------------------------------------------------------------
# Прогноз для дашборда (исторические данные + ARIMA forecast)
# ---------------------------------------------------------------------------

def get_skill_forecast(
    weekly_demand: pd.DataFrame,
    skill: str,
    n_forecast: int = 4,
) -> Dict:
    """Возвращает исторические данные + ARIMA-прогноз для страницы Forecast."""
    sk = (weekly_demand[weekly_demand['skill'] == skill]
          .sort_values(['year', 'week']))

    history_y     = sk['demand_index'].values
    history_weeks = sk['year_week'].values

    # Генерируем будущие метки недель (просто порядковые номера)
    future_labels = [f"W+{i+1}" for i in range(n_forecast)]

    fitted, order = fit_arima(history_y)
    if fitted is not None:
        try:
            fc = fitted.get_forecast(steps=n_forecast)
            forecast_mean = fc.predicted_mean
            ci = fc.conf_int(alpha=0.2)
            ci_lower = ci.iloc[:, 0].values
            ci_upper = ci.iloc[:, 1].values
        except Exception:
            forecast_mean = np.full(n_forecast, history_y[-1])
            ci_lower = forecast_mean * 0.9
            ci_upper = forecast_mean * 1.1
    else:
        forecast_mean = np.full(n_forecast, history_y[-1])
        ci_lower = forecast_mean * 0.9
        ci_upper = forecast_mean * 1.1

    return {
        'history_y':      history_y,
        'history_weeks':  history_weeks,
        'forecast_mean':  forecast_mean,
        'ci_lower':       ci_lower,
        'ci_upper':       ci_upper,
        'future_labels':  future_labels,
        'arima_order':    str(order),
    }
