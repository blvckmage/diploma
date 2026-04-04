#!/usr/bin/env python3
"""
Прогнозирование востребованности профессий на будущие периоды
Time Series Forecasting: предсказываем спрос через 1, 2, 3 месяца
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def create_future_features(df_monthly: pd.DataFrame, profession: str, last_months: int = 6) -> dict:
    """
    Создание признаков для прогнозирования будущего спроса
    
    Args:
        df_monthly: Данные агрегированные по месяцам
        profession: Профессия для прогноза
        last_months: Количество последних месяцев для анализа тренда
    
    Returns:
        Словарь с признаками для прогноза
    """
    prof_data = df_monthly[df_monthly['profession'] == profession].sort_values('month_num')
    
    if len(prof_data) < 3:
        return None
    
    last_data = prof_data.tail(last_months)
    
    features = {
        'profession_encoded': profession,
        'last_vacancy_count': last_data['vacancy_count'].iloc[-1],
        'avg_vacancy_last_3': last_data['vacancy_count'].tail(3).mean(),
        'avg_vacancy_last_6': last_data['vacancy_count'].mean(),
        'vacancy_trend': (last_data['vacancy_count'].iloc[-1] - last_data['vacancy_count'].iloc[0]) / (len(last_data) + 1),
        'vacancy_std': last_data['vacancy_count'].std(),
        'last_avg_salary': last_data['avg_salary'].iloc[-1] if 'avg_salary' in last_data.columns else 0,
        'last_employers': last_data['unique_employers'].iloc[-1] if 'unique_employers' in last_data.columns else 0,
        'last_cities': last_data['unique_cities'].iloc[-1] if 'unique_cities' in last_data.columns else 0,
        'months_data': len(prof_data),  # Сколько месяцев данных
        'max_vacancies': prof_data['vacancy_count'].max(),
        'min_vacancies': prof_data['vacancy_count'].min(),
    }
    
    # Месяц для сезонности
    last_month = prof_data['month_num'].iloc[-1]
    features['month'] = last_month % 12 + 1
    features['quarter'] = (last_month - 1) // 3 % 4 + 1
    
    return features


def prepare_training_data(df_monthly: pd.DataFrame, forecast_horizon: int = 1):
    """
    Подготовка обучающих данных для прогнозирования
    
    Args:
        df_monthly: Данные по месяцам
        forecast_horizon: На сколько месяцев вперёд прогнозируем (1, 2, 3)
    
    Returns:
        X, y, feature_columns
    """
    print(f"\n🔧 Подготовка данных для прогнозирования на {forecast_horizon} месяц(ев) вперёд...")
    
    # Кодирование профессий
    le = LabelEncoder()
    all_professions = df_monthly['profession'].unique()
    le.fit(all_professions)
    
    X_list = []
    y_list = []
    
    feature_columns = [
        'profession_encoded', 'last_vacancy_count', 'avg_vacancy_last_3', 
        'avg_vacancy_last_6', 'vacancy_trend', 'vacancy_std',
        'last_avg_salary', 'last_employers', 'last_cities',
        'month', 'quarter', 'months_data', 'max_vacancies', 'min_vacancies'
    ]
    
    for profession in all_professions:
        prof_data = df_monthly[df_monthly['profession'] == profession].sort_values('month_num')
        
        # Для каждого месяца (кроме последних forecast_horizon) создаём пример
        for i in range(len(prof_data) - forecast_horizon):
            current_data = prof_data.iloc[:i+1]
            future_data = prof_data.iloc[i+forecast_horizon]
            
            if len(current_data) < 3:
                continue
            
            # Признаки из текущих данных
            features = {
                'profession_encoded': le.transform([profession])[0],
                'last_vacancy_count': current_data['vacancy_count'].iloc[-1],
                'avg_vacancy_last_3': current_data['vacancy_count'].tail(3).mean(),
                'avg_vacancy_last_6': current_data['vacancy_count'].mean(),
                'vacancy_trend': (current_data['vacancy_count'].iloc[-1] - current_data['vacancy_count'].iloc[0]) / (len(current_data) + 1),
                'vacancy_std': current_data['vacancy_count'].std() if len(current_data) > 1 else 0,
                'last_avg_salary': current_data['avg_salary'].iloc[-1] if 'avg_salary' in current_data.columns else 0,
                'last_employers': current_data['unique_employers'].iloc[-1] if 'unique_employers' in current_data.columns else 0,
                'last_cities': current_data['unique_cities'].iloc[-1] if 'unique_cities' in current_data.columns else 0,
                'month': current_data['month_num'].iloc[-1] % 12 + 1,
                'quarter': (current_data['month_num'].iloc[-1] - 1) // 3 % 4 + 1,
                'months_data': len(current_data),
                'max_vacancies': current_data['vacancy_count'].max(),
                'min_vacancies': current_data['vacancy_count'].min(),
            }
            
            # Целевая переменная - количество вакансий через forecast_horizon месяцев
            target = future_data['vacancy_count']
            
            X_list.append(features)
            y_list.append(target)
    
    X = pd.DataFrame(X_list)
    y = pd.Series(y_list, name='target_vacancy_count')
    
    print(f"   ✅ Создано {len(X)} обучающих примеров")
    print(f"   📊 Признаков: {len(X.columns)}")
    print(f"   📅 Горизонт прогноза: {forecast_horizon} месяц(ев)")
    
    return X, y, feature_columns, le


def train_forecast_model(X, y):
    """
    Обучение модели прогнозирования
    
    Returns:
        Обученная модель, метрики
    """
    print("\n🤖 Обучение моделей прогнозирования...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    best_model = None
    best_score = -np.inf
    best_name = None
    
    print("\n📊 Сравнение моделей:")
    print("-" * 60)
    
    for name, model in models.items():
        print(f"\n🔧 {name}...")
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse
        }
        
        print(f"   Train R²: {train_r2:.4f}")
        print(f"   Test R²:  {test_r2:.4f}")
        print(f"   Test MAE: {test_mae:.2f}")
        print(f"   Test RMSE: {test_rmse:.2f}")
        
        if test_r2 > best_score:
            best_score = test_r2
            best_model = model
            best_name = name
    
    print(f"\n🏆 Лучшая модель: {best_name} (Test R² = {best_score:.4f})")
    
    return best_model, results, X_test, y_test


def forecast_future(df_monthly: pd.DataFrame, model, le: LabelEncoder, 
                     feature_columns: list, months_ahead: int = 3):
    """
    Прогнозирование востребованности на будущие периоды
    
    Args:
        df_monthly: Исторические данные
        model: Обученная модель
        le: LabelEncoder для профессий
        feature_columns: Список признаков
        months_ahead: На сколько месяцев вперёд
    
    Returns:
        DataFrame с прогнозами
    """
    print(f"\n🔮 Прогнозирование на {months_ahead} месяц(ев) вперёд...")
    
    all_professions = df_monthly['profession'].unique()
    forecasts = []
    
    for profession in all_professions:
        prof_data = df_monthly[df_monthly['profession'] == profession].sort_values('month_num')
        
        if len(prof_data) < 3:
            continue
        
        last_data = prof_data.tail(6)
        
        # Признаки для прогноза
        features = {
            'profession_encoded': le.transform([profession])[0],
            'last_vacancy_count': last_data['vacancy_count'].iloc[-1],
            'avg_vacancy_last_3': last_data['vacancy_count'].tail(3).mean(),
            'avg_vacancy_last_6': last_data['vacancy_count'].mean(),
            'vacancy_trend': (last_data['vacancy_count'].iloc[-1] - last_data['vacancy_count'].iloc[0]) / (len(last_data) + 1),
            'vacancy_std': last_data['vacancy_count'].std() if len(last_data) > 1 else 0,
            'last_avg_salary': last_data['avg_salary'].iloc[-1] if 'avg_salary' in last_data.columns else 0,
            'last_employers': last_data['unique_employers'].iloc[-1] if 'unique_employers' in last_data.columns else 0,
            'last_cities': last_data['unique_cities'].iloc[-1] if 'unique_cities' in last_data.columns else 0,
            'month': (last_data['month_num'].iloc[-1] % 12) + months_ahead,
            'quarter': ((last_data['month_num'].iloc[-1] + months_ahead - 1) // 3 % 4) + 1,
            'months_data': len(prof_data),
            'max_vacancies': prof_data['vacancy_count'].max(),
            'min_vacancies': prof_data['vacancy_count'].min(),
        }
        
        X_pred = pd.DataFrame([features])
        
        # Прогноз
        predicted_vacancies = model.predict(X_pred)[0]
        
        # Тренд
        current_vacancies = last_data['vacancy_count'].iloc[-1]
        avg_vacancies = last_data['vacancy_count'].mean()
        
        # Определение уровня востребованности
        if predicted_vacancies > avg_vacancies * 1.2:
            demand_level = 'Высокий'
        elif predicted_vacancies < avg_vacancies * 0.8:
            demand_level = 'Низкий'
        else:
            demand_level = 'Средний'
        
        forecasts.append({
            'profession': profession,
            'current_vacancies': int(current_vacancies),
            'predicted_vacancies': int(round(predicted_vacancies)),
            'avg_historical': int(round(avg_vacancies)),
            'demand_level': demand_level,
            'change_pct': round((predicted_vacancies - current_vacancies) / current_vacancies * 100, 1),
            'trend': '📈 Рост' if predicted_vacancies > current_vacancies else ('📉 Падение' if predicted_vacancies < current_vacancies else '➡️ Стабильно')
        })
    
    forecasts_df = pd.DataFrame(forecasts)
    forecasts_df = forecasts_df.sort_values('predicted_vacancies', ascending=False)
    
    return forecasts_df


def create_visualizations(forecasts_df: pd.DataFrame, results: dict, 
                          output_dir: str = 'reports/figures'):
    """
    Создание визуализаций прогнозов
    """
    print("\n📊 Создание визуализаций...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Топ профессии по прогнозируемому спросу
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    top_10 = forecasts_df.head(10)
    
    # Текущий vs Прогноз
    x = np.arange(len(top_10))
    width = 0.35
    
    axes[0].bar(x - width/2, top_10['current_vacancies'], width, label='Текущий', color='steelblue')
    axes[0].bar(x + width/2, top_10['predicted_vacancies'], width, label='Прогноз', color='coral')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(top_10['profession'], rotation=45, ha='right')
    axes[0].set_ylabel('Количество вакансий')
    axes[0].set_title('Топ-10 профессий: текущий vs прогноз')
    axes[0].legend()
    
    # Изменение в %
    colors = ['green' if x > 0 else 'red' for x in forecasts_df['change_pct'].head(10)]
    axes[1].barh(forecasts_df['profession'].head(10), forecasts_df['change_pct'].head(10), color=colors)
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Изменение спроса (%)')
    axes[1].set_title('Прогнозируемое изменение спроса')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/forecast_top_professions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_dir}/forecast_top_professions.png")
    
    # 2. Распределение по уровням востребованности
    fig, ax = plt.subplots(figsize=(10, 6))
    
    demand_counts = forecasts_df['demand_level'].value_counts()
    colors = {'Высокий': 'green', 'Средний': 'gold', 'Низкий': 'red'}
    
    ax.bar(demand_counts.index, demand_counts.values, 
           color=[colors.get(x, 'gray') for x in demand_counts.index])
    ax.set_xlabel('Уровень востребованности')
    ax.set_ylabel('Количество профессий')
    ax.set_title('Прогноз востребованности профессий')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/forecast_demand_levels.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_dir}/forecast_demand_levels.png")
    
    # 3. Сравнение моделей
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(results.keys())
    test_r2 = [results[n]['test_r2'] for n in model_names]
    test_mae = [results[n]['test_mae'] for n in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width/2, test_r2, width, label='Test R²', color='steelblue')
    bars2 = ax2.bar(x + width/2, test_mae, width, label='Test MAE', color='coral')
    
    ax.set_ylabel('R² Score', color='steelblue')
    ax2.set_ylabel('MAE (вакансии)', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_title('Сравнение моделей прогнозирования')
    
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/forecast_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_dir}/forecast_model_comparison.png")


def run_forecast_pipeline(data_path: str = 'results_synthetic.csv', 
                          months_ahead: int = 3):
    """
    Запуск пайплайна прогнозирования
    
    Args:
        data_path: Путь к данным
        months_ahead: На сколько месяцев вперёд прогнозировать
    """
    print("=" * 70)
    print("🔮 ПРОГНОЗИРОВАНИЕ ВОСТРЕБОВАННОСТИ ПРОФЕССИЙ")
    print("=" * 70)
    print(f"📅 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Горизонт прогноза: {months_ahead} месяц(ев)")
    print()
    
    # ===== ЭТАП 1: ЗАГРУЗКА ДАННЫХ =====
    print("=" * 50)
    print("📊 ЭТАП 1: ЗАГРУЗКА И АГРЕГАЦИЯ ДАННЫХ")
    print("=" * 50)
    
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    print(f"✅ Загружено {len(df)} вакансий")
    
    # Парсинг даты
    df['publish_date'] = pd.to_datetime(df['publish_date'], format='%d-%m-%Y', errors='coerce')
    
    # Агрегация по месяцам
    df['year'] = df['publish_date'].dt.year
    df['month'] = df['publish_date'].dt.month
    df['month_num'] = (df['year'] - df['year'].min()) * 12 + df['month']
    
    # Парсинг зарплат
    def parse_salary(s):
        if pd.isna(s) or s == '':
            return np.nan
        import re
        nums = re.findall(r'\d+', str(s))
        return np.mean([int(n) for n in nums]) if nums else np.nan
    
    df['salary_numeric'] = df['salary'].apply(parse_salary)
    
    # Агрегация по месяцам и профессиям
    df_monthly = df.groupby(['month_num', 'Job']).agg({
        'title': 'count',
        'salary_numeric': 'mean',
        'city': 'nunique',
        'job': 'nunique'
    }).reset_index()
    
    df_monthly.columns = ['month_num', 'profession', 'vacancy_count', 'avg_salary', 
                          'unique_cities', 'unique_employers']
    
    print(f"✅ Агрегировано по месяцам: {len(df_monthly)} записей")
    print(f"   📊 Профессий: {df_monthly['profession'].nunique()}")
    print(f"   📅 Месяцев: {df_monthly['month_num'].nunique()}")
    
    # ===== ЭТАП 2: ПОДГОТОВКА ДАННЫХ =====
    print("\n" + "=" * 50)
    print("🔧 ЭТАП 2: ПОДГОТОВКА ДАННЫХ ДЛЯ ПРОГНОЗИРОВАНИЯ")
    print("=" * 50)
    
    X, y, feature_cols, le = prepare_training_data(df_monthly, forecast_horizon=months_ahead)
    
    # ===== ЭТАП 3: ОБУЧЕНИЕ =====
    print("\n" + "=" * 50)
    print("🤖 ЭТАП 3: ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 50)
    
    best_model, results, X_test, y_test = train_forecast_model(X, y)
    
    # ===== ЭТАП 4: ПРОГНОЗИРОВАНИЕ =====
    print("\n" + "=" * 50)
    print("🔮 ЭТАП 4: ПРОГНОЗИРОВАНИЕ БУДУЩЕГО СПРОСА")
    print("=" * 50)
    
    forecasts_df = forecast_future(df_monthly, best_model, le, feature_cols, 
                                   months_ahead=months_ahead)
    
    print("\n📊 ПРОГНОЗ ВОСТРЕБОВАННОСТИ ПРОФЕССИЙ:")
    print(f"   Горизонт: {months_ahead} месяц(ев)")
    print()
    print(forecasts_df.to_string(index=False))
    
    # ===== ЭТАП 5: ВИЗУАЛИЗАЦИИ =====
    print("\n" + "=" * 50)
    print("📊 ЭТАП 5: СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("=" * 50)
    
    create_visualizations(forecasts_df, results)
    
    # ===== ЭТАП 6: СОХРАНЕНИЕ =====
    print("\n" + "=" * 50)
    print("💾 ЭТАП 6: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 50)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    joblib.dump(best_model, 'models/forecast_model.joblib')
    joblib.dump(le, 'models/profession_encoder.joblib')
    print("   ✅ Model saved: models/forecast_model.joblib")
    
    forecasts_df.to_csv('reports/forecasts.csv', index=False)
    print("   ✅ Forecasts saved: reports/forecasts.csv")
    
    # ===== ИТОГИ =====
    print("\n" + "=" * 70)
    print("✅ ПРОГНОЗИРОВАНИЕ ЗАВЕРШЕНО!")
    print("=" * 70)
    
    print(f"\n📊 Результаты прогноза на {months_ahead} месяц(ев) вперёд:")
    
    high_demand = forecasts_df[forecasts_df['demand_level'] == 'Высокий']['profession'].tolist()
    print(f"\n🚀 Профессии с ВЫСОКИМ спросом:")
    for prof in high_demand[:5]:
        row = forecasts_df[forecasts_df['profession'] == prof].iloc[0]
        print(f"   • {prof}: {row['predicted_vacancies']} вакансий ({row['trend']})")
    
    print(f"\n📁 Сохранённые файлы:")
    print(f"   • reports/forecasts.csv")
    print(f"   • reports/figures/forecast_top_professions.png")
    print(f"   • reports/figures/forecast_demand_levels.png")
    print(f"   • reports/figures/forecast_model_comparison.png")
    print(f"   • models/forecast_model.joblib")
    
    print(f"\n📅 Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return forecasts_df, best_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Прогнозирование востребованности профессий')
    parser.add_argument('--data', type=str, default='results_synthetic.csv',
                        help='Путь к файлу с данными')
    parser.add_argument('--months', type=int, default=3,
                        help='На сколько месяцев вперёд прогнозировать')
    
    args = parser.parse_args()
    
    run_forecast_pipeline(data_path=args.data, months_ahead=args.months)