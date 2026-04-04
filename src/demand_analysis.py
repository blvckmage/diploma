#!/usr/bin/env python3
"""
Модуль анализа востребованности профессий
Агрегация данных и создание целевой переменной для прогнозирования спроса
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DemandAnalyzer:
    """Класс для анализа и прогнозирования востребованности профессий"""
    
    # Веса для комплексного индекса востребованности
    DEMAND_WEIGHTS = {
        'vacancy_count': 0.30,      # Количество вакансий
        'salary_level': 0.25,       # Уровень зарплат
        'employer_diversity': 0.15, # Разнообразие работодателей
        'geo_coverage': 0.10,       # Географический охват
        'growth_trend': 0.10,       # Тренд роста
        'competition': 0.10         # Конкуренция (обратный показатель)
    }
    
    def __init__(self):
        self.df = None
        self.weekly_demand = None
        self.profession_stats = None
        self.demand_index_stats = None
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Загрузка данных о вакансиях"""
        self.df = pd.read_csv(filepath, encoding='utf-8-sig')
        
        # Парсинг даты
        self.df['publish_date'] = pd.to_datetime(
            self.df['publish_date'], 
            format='%d-%m-%Y', 
            errors='coerce'
        )
        
        print(f"✅ Загружено {len(self.df)} вакансий")
        print(f"   Период: {self.df['publish_date'].min()} - {self.df['publish_date'].max()}")
        
        return self.df
    
    def aggregate_weekly_demand(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Агрегация данных по неделям и профессиям
        
        Returns:
            DataFrame с колонками: week, profession, vacancy_count, avg_salary, ...
        """
        if df is not None:
            self.df = df
        
        if self.df is None:
            raise ValueError("Данные не загружены")
        
        print("🔄 Агрегация данных по неделям и профессиям...")
        
        # Добавляем неделю
        self.df['week'] = self.df['publish_date'].dt.isocalendar().week
        self.df['year'] = self.df['publish_date'].dt.year
        self.df['year_week'] = self.df['year'].astype(str) + '_W' + self.df['week'].astype(str).str.zfill(2)
        
        # Парсинг зарплат
        self.df['salary_numeric'] = self.df['salary'].apply(self._parse_salary)
        
        # Агрегация по неделе и профессии
        agg_data = []
        
        for (year_week, profession), group in self.df.groupby(['year_week', 'Job']):
            vacancy_count = len(group)
            avg_salary = group['salary_numeric'].mean()
            cities = group['city'].nunique()
            employers = group['job'].nunique()
            
            # Определяем неделю для сортировки
            year, week = map(int, year_week.replace('W', '').split('_'))
            
            agg_data.append({
                'year_week': year_week,
                'year': year,
                'week': week,
                'profession': profession,
                'vacancy_count': vacancy_count,
                'avg_salary': avg_salary,
                'unique_cities': cities,
                'unique_employers': employers,
                'has_salary_ratio': group['salary_numeric'].notna().mean()
            })
        
        self.weekly_demand = pd.DataFrame(agg_data)
        self.weekly_demand = self.weekly_demand.sort_values(['year', 'week', 'profession'])
        
        print(f"   ✅ Создано {len(self.weekly_demand)} записей")
        print(f"   📊 {self.weekly_demand['profession'].nunique()} профессий")
        print(f"   📅 {self.weekly_demand['year_week'].nunique()} недель")
        
        return self.weekly_demand
    
    def _parse_salary(self, salary_str) -> float:
        """Парсинг зарплаты из строки"""
        if pd.isna(salary_str) or str(salary_str).strip() == '':
            return np.nan
        
        import re
        numbers = re.findall(r'\d+', str(salary_str))
        if not numbers:
            return np.nan
        
        return np.mean([int(n) for n in numbers])
    
    def calculate_demand_index(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Вычисление комплексного индекса востребованности
        
        Формула:
        Demand_Index = Σ(w_i * norm(feature_i))
        
        Компоненты:
        1. vacancy_score (30%) - нормализованное количество вакансий
        2. salary_score (25%) - уровень зарплат относительно рынка
        3. employer_score (15%) - разнообразие работодателей
        4. geo_score (10%) - географический охват
        5. growth_score (10%) - тренд роста вакансий
        6. competition_score (10%) - обратная конкуренция
        
        Returns:
            DataFrame с calculated demand_index и компонентами
        """
        if df is not None:
            self.weekly_demand = df
        
        if self.weekly_demand is None:
            raise ValueError("Сначала выполните aggregate_weekly_demand()")
        
        print("🔄 Вычисление комплексного индекса востребованности...")
        
        df = self.weekly_demand.copy()
        
        # 1. Vacancy Score - нормализация количества вакансий (min-max)
        max_vacancies = df['vacancy_count'].max()
        min_vacancies = df['vacancy_count'].min()
        df['vacancy_score'] = (df['vacancy_count'] - min_vacancies) / (max_vacancies - min_vacancies + 1)
        
        # 2. Salary Score - уровень зарплат относительно медианы рынка
        median_salary = df['avg_salary'].median()
        df['salary_score'] = df['avg_salary'] / (median_salary + 1)
        df['salary_score'] = df['salary_score'].clip(0, 2) / 2  # Нормализация [0, 1]
        df['salary_score'] = df['salary_score'].fillna(0.5)
        
        # 3. Employer Score - разнообразие работодателей (log scale)
        df['employer_score'] = np.log1p(df['unique_employers']) / np.log1p(df['unique_employers'].max() + 1)
        df['employer_score'] = df['employer_score'].fillna(0)
        
        # 4. Geo Score - географический охват
        max_cities = df['unique_cities'].max()
        df['geo_score'] = df['unique_cities'] / (max_cities + 1)
        df['geo_score'] = df['geo_score'].fillna(0)
        
        # 5. Growth Score - вычисляем тренд роста
        df = df.sort_values(['profession', 'year', 'week'])
        df['prev_vacancy_count'] = df.groupby('profession')['vacancy_count'].shift(1)
        df['growth_rate'] = (df['vacancy_count'] - df['prev_vacancy_count']) / (df['prev_vacancy_count'] + 1)
        df['growth_score'] = (df['growth_rate'] + 1) / 2  # Нормализация [-1,1] -> [0,1]
        df['growth_score'] = df['growth_score'].clip(0, 1).fillna(0.5)
        
        # 6. Competition Score - обратный показатель (меньше работодателей на вакансию = больше конкуренция)
        df['vacancy_per_employer'] = df['vacancy_count'] / (df['unique_employers'] + 1)
        max_vpe = df['vacancy_per_employer'].max()
        df['competition_score'] = 1 - (df['vacancy_per_employer'] / (max_vpe + 1))
        df['competition_score'] = df['competition_score'].clip(0, 1).fillna(0.5)
        
        # === КОМПЛЕКСНЫЙ ИНДЕКС ===
        df['demand_index'] = (
            self.DEMAND_WEIGHTS['vacancy_count'] * df['vacancy_score'] +
            self.DEMAND_WEIGHTS['salary_level'] * df['salary_score'] +
            self.DEMAND_WEIGHTS['employer_diversity'] * df['employer_score'] +
            self.DEMAND_WEIGHTS['geo_coverage'] * df['geo_score'] +
            self.DEMAND_WEIGHTS['growth_trend'] * df['growth_score'] +
            self.DEMAND_WEIGHTS['competition'] * df['competition_score']
        )
        
        # Очистка временных колонок
        df = df.drop(columns=['prev_vacancy_count', 'growth_rate', 'vacancy_per_employer'], errors='ignore')
        
        self.weekly_demand = df
        
        # Статистика индекса
        print(f"   📊 Индекс востребованности:")
        print(f"      Min: {df['demand_index'].min():.4f}")
        print(f"      Max: {df['demand_index'].max():.4f}")
        print(f"      Mean: {df['demand_index'].mean():.4f}")
        print(f"      Std: {df['demand_index'].std():.4f}")
        
        return df
    
    def create_demand_target(self, df: Optional[pd.DataFrame] = None, n_classes: int = 5) -> pd.DataFrame:
        """
        Создание целевой переменной на основе комплексного индекса
        
        Классификация по квантилям (adaptive thresholds):
        - 5 классов: Очень низкий, Низкий, Средний, Высокий, Очень высокий
        - 3 класса: Низкий, Средний, Высокий
        
        Args:
            df: DataFrame с данными
            n_classes: Количество классов (3 или 5)
        
        Returns:
            DataFrame с целевой переменной 'demand_level'
        """
        if df is not None:
            self.weekly_demand = df
        
        if self.weekly_demand is None:
            raise ValueError("Сначала выполните aggregate_weekly_demand()")
        
        # Сначала вычисляем индекс
        if 'demand_index' not in self.weekly_demand.columns:
            self.calculate_demand_index()
        
        print(f"🔄 Создание целевой переменной ({n_classes} классов)...")
        
        df = self.weekly_demand
        
        # Классификация по квантилям
        if n_classes == 5:
            # 5 классов по квантилям: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
            quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = {
                0: 'Очень низкий',
                1: 'Низкий', 
                2: 'Средний',
                3: 'Высокий',
                4: 'Очень высокий'
            }
        else:  # n_classes == 3
            quantiles = [0, 0.33, 0.67, 1.0]
            labels = {
                0: 'Низкий',
                1: 'Средний',
                2: 'Высокий'
            }
        
        # Вычисляем пороги
        thresholds = df['demand_index'].quantile(quantiles).values
        
        # Классификация
        df['demand_level'] = pd.cut(
            df['demand_index'],
            bins=thresholds,
            labels=list(labels.keys()),
            include_lowest=True
        ).astype(int)
        
        self.weekly_demand = df
        self.demand_index_stats = {
            'thresholds': thresholds,
            'labels': labels,
            'n_classes': n_classes
        }
        
        # Статистика
        demand_dist = df['demand_level'].value_counts().sort_index()
        print(f"   📊 Распределение классов:")
        for level, count in demand_dist.items():
            label = labels.get(level, f'Класс {level}')
            pct = count / len(df) * 100
            print(f"      {label} ({level}): {count} записей ({pct:.1f}%)")
        
        return df
    
    def get_demand_components_breakdown(self, profession: str = None) -> pd.DataFrame:
        """
        Получение детального разбора компонентов индекса востребованности
        
        Args:
            profession: Конкретная профессия (если None, то все)
        
        Returns:
            DataFrame с компонентами индекса
        """
        if self.weekly_demand is None or 'demand_index' not in self.weekly_demand.columns:
            raise ValueError("Сначала выполните calculate_demand_index()")
        
        df = self.weekly_demand
        
        if profession:
            df = df[df['profession'] == profession]
        
        components = ['profession', 'vacancy_count', 'demand_index',
                      'vacancy_score', 'salary_score', 'employer_score',
                      'geo_score', 'growth_score', 'competition_score']
        
        return df[components].copy()
    
    def create_features_for_demand(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Создание признаков для прогнозирования востребованности
        
        Features:
        - Временные: месяц, квартал, неделя года
        - Профессия: encoded profession
        - Исторические: скользящие средние, лаги
        - Рыночные: зарплата, работодатели
        """
        if df is not None:
            self.weekly_demand = df
        
        if self.weekly_demand is None:
            raise ValueError("Сначала выполните aggregate_weekly_demand()")
        
        print("🔧 Создание признаков для прогнозирования спроса...")
        
        df = self.weekly_demand.copy()
        
        # 1. Временные признаки
        print("   • Временные признаки...")
        df['month'] = df['week'] // 4 + 1
        df['quarter'] = df['week'] // 13 + 1
        df['is_month_start'] = (df['week'] % 4 == 1).astype(int)
        df['is_month_end'] = (df['week'] % 4 == 0).astype(int)
        
        # 2. Кодирование профессии
        print("   • Кодирование профессий...")
        from sklearn.preprocessing import LabelEncoder
        le_prof = LabelEncoder()
        df['profession_encoded'] = le_prof.fit_transform(df['profession'].astype(str))
        
        # 3. Исторические признаки (лаги и скользящие средние)
        print("   • Исторические признаки...")
        df = df.sort_values(['profession', 'year', 'week'])
        
        # Лаги только для vacancy_count (НЕ demand_level - это утечка!)
        for lag in [1, 2, 3, 4]:
            df[f'vacancy_count_lag{lag}'] = df.groupby('profession')['vacancy_count'].shift(lag)
        
        # Скользящие средние
        for window in [2, 4, 8]:
            df[f'vacancy_count_rolling_{window}'] = df.groupby('profession')['vacancy_count'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # 4. Статистика по профессии
        print("   • Статистика по профессиям...")
        prof_stats = df.groupby('profession').agg({
            'vacancy_count': ['mean', 'std', 'max', 'min'],
            'avg_salary': 'mean',
            'unique_employers': 'mean'
        }).reset_index()
        
        prof_stats.columns = ['profession', 'prof_avg_vacancies', 'prof_std_vacancies', 
                              'prof_max_vacancies', 'prof_min_vacancies',
                              'prof_avg_salary', 'prof_avg_employers']
        
        df = df.merge(prof_stats, on='profession', how='left')
        
        # 5. Тренд (изменение относительно предыдущей недели)
        print("   • Признаки тренда...")
        df['vacancy_growth'] = df.groupby('profession')['vacancy_count'].pct_change()
        df['is_growing'] = (df['vacancy_growth'] > 0).astype(int)
        
        # 6. Заполнение пропусков
        df = df.fillna(0)
        
        self.weekly_demand = df
        
        print(f"   ✅ Создано признаков: {len(df.columns)}")
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Получение списка колонок-признаков для ML
        
        ВАЖНО: Исключаем компоненты индекса для предотвращения утечки данных!
        """
        # Колонки, которые нельзя использовать (прямая утечка)
        exclude_cols = [
            'year_week',           # Идентификатор
            'profession',          # Категориальный (используем encoded)
            'demand_level',        # Целевая переменная
            'vacancy_count',       # Компонент индекса
            'demand_index',        # Сам индекс (утечка!)
            'vacancy_score',       # Компонент индекса - УТЕЧКА
            'salary_score',        # Компонент индекса - УТЕЧКА
            'employer_score',      # Компонент индекса - УТЕЧКА
            'geo_score',           # Компонент индекса - УТЕЧКА
            'growth_score',        # Компонент индекса - УТЕЧКА
            'competition_score',   # Компонент индекса - УТЕЧКА
        ]
        feature_cols = [col for col in self.weekly_demand.columns 
                        if col not in exclude_cols]
        return feature_cols
    
    def prepare_for_modeling(self, n_classes: int = 5) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Полная подготовка данных для моделирования
        
        Args:
            n_classes: Количество классов (3 или 5)
        
        Returns:
            X, y, feature_columns
        """
        # Агрегация
        self.aggregate_weekly_demand()
        
        # Расчёт комплексного индекса
        self.calculate_demand_index()
        
        # Целевая переменная
        self.create_demand_target(n_classes=n_classes)
        
        # Признаки
        self.create_features_for_demand()
        
        # Выбор признаков
        feature_cols = self.get_feature_columns()
        
        X = self.weekly_demand[feature_cols]
        y = self.weekly_demand['demand_level']
        
        # Определяем названия классов
        if n_classes == 5:
            level_names = {0: 'Очень низкий', 1: 'Низкий', 2: 'Средний', 3: 'Высокий', 4: 'Очень высокий'}
        else:
            level_names = {0: 'Низкий', 1: 'Средний', 2: 'Высокий'}
        
        print("\n" + "=" * 60)
        print("✅ ДАННЫЕ ГОТОВЫ ДЛЯ МОДЕЛИРОВАНИЯ")
        print("=" * 60)
        print(f"   • Записей: {len(X)}")
        print(f"   • Признаков: {len(feature_cols)}")
        print(f"   • Классов: {y.nunique()}")
        print(f"   • Распределение классов:")
        for level, count in y.value_counts().sort_index().items():
            print(f"      {level_names.get(level, f'Класс {level}')}: {count} ({count/len(y)*100:.1f}%)")
        print("=" * 60)
        
        return X, y, feature_cols
    
    def analyze_demand_trends(self) -> pd.DataFrame:
        """Анализ трендов востребованности по профессиям"""
        if self.weekly_demand is None:
            raise ValueError("Сначала выполните aggregate_weekly_demand()")
        
        trends = self.weekly_demand.groupby('profession').agg({
            'vacancy_count': ['sum', 'mean', 'std'],
            'avg_salary': 'mean',
            'demand_level': lambda x: (x == 2).sum()  # Количество недель с высоким спросом
        }).reset_index()
        
        trends.columns = ['profession', 'total_vacancies', 'avg_weekly_vacancies', 
                          'std_weekly_vacancies', 'avg_salary', 'high_demand_weeks']
        
        # Сортировка по популярности
        trends = trends.sort_values('total_vacancies', ascending=False)
        
        return trends
    
    def get_most_sought_after_professions(self, top_n: int = 10) -> List[str]:
        """Получение топ-N самых востребованных профессий"""
        trends = self.analyze_demand_trends()
        return trends.head(top_n)['profession'].tolist()


if __name__ == "__main__":
    # Тестирование
    analyzer = DemandAnalyzer()
    analyzer.load_data('results.csv')
    
    X, y, features = analyzer.prepare_for_modeling()
    
    print("\n📊 Топ востребованные профессии:")
    top_professions = analyzer.get_most_sought_after_professions(10)
    for i, prof in enumerate(top_professions, 1):
        print(f"   {i}. {prof}")
