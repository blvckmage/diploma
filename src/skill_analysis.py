#!/usr/bin/env python3
"""
Модуль анализа востребованности навыков
Агрегация данных и создание целевой переменной для прогнозирования спроса на ИТ-навыки
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.preprocessing import VacancyPreprocessor
from src.features import FeatureEngineer

class SkillAnalyzer:
    """Класс для анализа и прогнозирования востребованности ИТ-навыков"""
    
    # Веса для комплексного индекса востребованности
    DEMAND_WEIGHTS = {
        'vacancy_count': 0.30,      # Количество вакансий с навыком
        'salary_level': 0.25,       # Уровень зарплат для навыка
        'employer_diversity': 0.15, # Разнообразие работодателей
        'geo_coverage': 0.10,       # Географический охват
        'growth_trend': 0.10,       # Тренд роста
        'competition': 0.10         # Конкуренция
    }
    
    def __init__(self):
        self.df = None
        self.exploded_df = None
        self.weekly_demand = None
        self.demand_index_stats = None
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Загрузка и базовая предобработка данных о вакансиях"""
        preprocessor = VacancyPreprocessor()
        self.df = preprocessor.load_data(filepath)
        self.df = preprocessor.preprocess(self.df)
        
        print(f"✅ Данные загружены и предобработаны: {len(self.df)} вакансий")
        return self.df

    def extract_and_explode_skills(self, df: Optional[pd.DataFrame] = None, min_frequency: int = 10) -> pd.DataFrame:
        """
        Извлечение навыков и "взрыв" датафрейма (одна строка = один навык из вакансии)
        
        Args:
            min_frequency: Минимальное количество упоминаний навыка для включения
        """
        if df is not None:
            self.df = df
            
        if self.df is None:
            raise ValueError("Данные не загружены")
            
        print("🔄 Извлечение навыков и трансформация данных...")
        
        fe = FeatureEngineer()
        
        # Извлекаем навыки
        self.df['skills_extracted'] = self.df['clean_text'].apply(fe.extract_skills)
        
        # Оставляем только нужные колонки для ускорения
        cols_to_keep = ['index', 'publish_date', 'salary_numeric', 'city', 'job', 'Job', 'skills_extracted']
        cols_to_keep = [c for c in cols_to_keep if c in self.df.columns]
        
        subset_df = self.df[cols_to_keep].copy()
        
        # Explode: 1 строка вакансии с N навыками -> N строк (по одной на навык)
        exploded = subset_df.explode('skills_extracted')
        exploded = exploded.rename(columns={'skills_extracted': 'skill'})
        
        # Удаляем пустые навыки
        exploded = exploded.dropna(subset=['skill'])
        
        # Фильтрация редких навыков
        skill_counts = exploded['skill'].value_counts()
        frequent_skills = skill_counts[skill_counts >= min_frequency].index
        exploded = exploded[exploded['skill'].isin(frequent_skills)]
        
        self.exploded_df = exploded
        
        print(f"   ✅ Трансформация завершена: {len(self.exploded_df)} пар (вакансия-навык)")
        print(f"   📊 Найдено уникальных навыков: {self.exploded_df['skill'].nunique()} (минимум {min_frequency} упоминаний)")
        
        return self.exploded_df
        
    def aggregate_weekly_skills(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Агрегация данных по неделям и навыкам
        """
        if df is not None:
            self.exploded_df = df
            
        if self.exploded_df is None:
            self.extract_and_explode_skills()
            
        print("🔄 Агрегация данных по неделям и навыкам...")
        
        # Добавляем неделю
        self.exploded_df['week'] = self.exploded_df['publish_date'].dt.isocalendar().week
        self.exploded_df['year'] = self.exploded_df['publish_date'].dt.year
        self.exploded_df['year_week'] = self.exploded_df['year'].astype(str) + '_W' + self.exploded_df['week'].astype(str).str.zfill(2)
        
        agg_data = []
        
        for (year_week, skill), group in self.exploded_df.groupby(['year_week', 'skill']):
            vacancy_count = len(group)
            avg_salary = group['salary_numeric'].mean()
            cities = group['city'].nunique()
            employers = group['job'].nunique()
            
            year, week = map(int, year_week.replace('W', '').split('_'))
            
            agg_data.append({
                'year_week': year_week,
                'year': year,
                'week': week,
                'skill': skill,
                'vacancy_count': vacancy_count,
                'avg_salary': avg_salary,
                'unique_cities': cities,
                'unique_employers': employers,
                'has_salary_ratio': group['salary_numeric'].notna().mean()
            })
            
        self.weekly_demand = pd.DataFrame(agg_data)
        self.weekly_demand = self.weekly_demand.sort_values(['year', 'week', 'skill'])
        
        print(f"   ✅ Создано {len(self.weekly_demand)} записей (неделя-навык)")
        
        return self.weekly_demand
        
    def calculate_demand_index(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Вычисление комплексного индекса востребованности навыка"""
        if df is not None:
            self.weekly_demand = df
            
        if self.weekly_demand is None:
            raise ValueError("Сначала выполните aggregate_weekly_skills()")
            
        print("🔄 Вычисление комплексного индекса востребованности...")
        df = self.weekly_demand.copy()
        
        # 1. Vacancy Score
        max_vacancies = df['vacancy_count'].max()
        min_vacancies = df['vacancy_count'].min()
        df['vacancy_score'] = (df['vacancy_count'] - min_vacancies) / (max_vacancies - min_vacancies + 1)
        
        # 2. Salary Score
        median_salary = df['avg_salary'].median()
        df['salary_score'] = df['avg_salary'] / (median_salary + 1)
        df['salary_score'] = df['salary_score'].clip(0, 2) / 2
        df['salary_score'] = df['salary_score'].fillna(0.5)
        
        # 3. Employer Score 
        df['employer_score'] = np.log1p(df['unique_employers']) / np.log1p(df['unique_employers'].max() + 1)
        df['employer_score'] = df['employer_score'].fillna(0)
        
        # 4. Geo Score
        max_cities = df['unique_cities'].max()
        df['geo_score'] = df['unique_cities'] / (max_cities + 1)
        df['geo_score'] = df['geo_score'].fillna(0)
        
        # 5. Growth Score 
        df = df.sort_values(['skill', 'year', 'week'])
        df['prev_vacancy_count'] = df.groupby('skill')['vacancy_count'].shift(1)
        df['growth_rate'] = (df['vacancy_count'] - df['prev_vacancy_count']) / (df['prev_vacancy_count'] + 1)
        df['growth_score'] = (df['growth_rate'] + 1) / 2
        df['growth_score'] = df['growth_score'].clip(0, 1).fillna(0.5)
        
        # 6. Competition Score
        df['vacancy_per_employer'] = df['vacancy_count'] / (df['unique_employers'] + 1)
        max_vpe = df['vacancy_per_employer'].max()
        df['competition_score'] = 1 - (df['vacancy_per_employer'] / (max_vpe + 1))
        df['competition_score'] = df['competition_score'].clip(0, 1).fillna(0.5)
        
        # Комплексный индекс
        df['demand_index'] = (
            self.DEMAND_WEIGHTS['vacancy_count'] * df['vacancy_score'] +
            self.DEMAND_WEIGHTS['salary_level'] * df['salary_score'] +
            self.DEMAND_WEIGHTS['employer_diversity'] * df['employer_score'] +
            self.DEMAND_WEIGHTS['geo_coverage'] * df['geo_score'] +
            self.DEMAND_WEIGHTS['growth_trend'] * df['growth_score'] +
            self.DEMAND_WEIGHTS['competition'] * df['competition_score']
        )
        
        df = df.drop(columns=['prev_vacancy_count', 'growth_rate', 'vacancy_per_employer'], errors='ignore')
        self.weekly_demand = df
        
        print(f"   📊 Индекс востребованности навыков: Mean: {df['demand_index'].mean():.4f}")
        return df
        
    def create_demand_target(self, df: Optional[pd.DataFrame] = None, n_classes: int = 5) -> pd.DataFrame:
        """Создание целевой переменной 'demand_level' по квантилям"""
        if df is not None:
            self.weekly_demand = df
        if self.weekly_demand is None:
            raise ValueError("Сначала выполните calculate_demand_index()")
            
        print(f"🔄 Создание целевой переменной ({n_classes} классов)...")
        df = self.weekly_demand
        
        if 'demand_index' not in df.columns:
            self.calculate_demand_index()
            
        if n_classes == 5:
            quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = {0: 'Очень низкий', 1: 'Низкий', 2: 'Средний', 3: 'Высокий', 4: 'Очень высокий'}
        else:
            quantiles = [0, 0.33, 0.67, 1.0]
            labels = {0: 'Низкий', 1: 'Средний', 2: 'Высокий'}
            
        thresholds = df['demand_index'].quantile(quantiles).values
        
        df['demand_level'] = pd.cut(
            df['demand_index'],
            bins=thresholds,
            labels=list(labels.keys()),
            include_lowest=True
        ).astype(int)
        
        self.weekly_demand = df
        self.demand_index_stats = {'thresholds': thresholds, 'labels': labels, 'n_classes': n_classes}
        
        return df

    def create_features_for_demand(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Создание ML признаков"""
        if df is not None:
            self.weekly_demand = df
        if self.weekly_demand is None:
            raise ValueError("Сначала выполните aggregate_weekly_skills()")
            
        print("🔧 Создание признаков для прогнозирования спроса на навыки...")
        df = self.weekly_demand.copy()
        
        # 1. Временные
        df['month'] = df['week'] // 4 + 1
        df['quarter'] = df['week'] // 13 + 1
        
        # 2. Кодирование навыка
        from sklearn.preprocessing import LabelEncoder
        le_skill = LabelEncoder()
        df['skill_encoded'] = le_skill.fit_transform(df['skill'].astype(str))
        
        # 3. Исторические (лаги и скользящие)
        df = df.sort_values(['skill', 'year', 'week'])
        for lag in [1, 2, 3, 4]:
            df[f'vacancy_count_lag{lag}'] = df.groupby('skill')['vacancy_count'].shift(lag)
            
        for window in [2, 4, 8]:
            df[f'vacancy_count_rolling_{window}'] = df.groupby('skill')['vacancy_count'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
        # 4. Статистика по навыку
        skill_stats = df.groupby('skill').agg({
            'vacancy_count': ['mean', 'std', 'max', 'min'],
            'avg_salary': 'mean',
            'unique_employers': 'mean'
        }).reset_index()
        
        skill_stats.columns = ['skill', 'skill_avg_vacancies', 'skill_std_vacancies', 
                               'skill_max_vacancies', 'skill_min_vacancies',
                               'skill_avg_salary', 'skill_avg_employers']
        
        df = df.merge(skill_stats, on='skill', how='left')
        
        # 5. Тренд
        df['vacancy_growth'] = df.groupby('skill')['vacancy_count'].pct_change()
        df['is_growing'] = (df['vacancy_growth'] > 0).astype(int)
        
        df = df.fillna(0)
        self.weekly_demand = df
        print(f"   ✅ Создано признаков: {len(df.columns)}")
        
        return df
        
    def get_feature_columns(self) -> List[str]:
        exclude_cols = [
            'year_week', 'skill', 'demand_level', 'vacancy_count', 
            'demand_index', 'vacancy_score', 'salary_score', 'employer_score', 
            'geo_score', 'growth_score', 'competition_score'
        ]
        return [col for col in self.weekly_demand.columns if col not in exclude_cols]
        
    def prepare_for_modeling(self, task: str = 'regression', n_classes: int = 5) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Полная подготовка: агрегация -> индекс -> таргет -> фичи -> X, y"""
        self.aggregate_weekly_skills()
        self.calculate_demand_index()
        self.create_demand_target(n_classes=n_classes)
        self.create_features_for_demand()
        
        feature_cols = self.get_feature_columns()
        X = self.weekly_demand[feature_cols]
        
        if task == 'regression':
            y = self.weekly_demand['demand_index']
        else:
            y = self.weekly_demand['demand_level']
        
        print("\n" + "=" * 60)
        print("✅ ДАННЫЕ О НАВЫКАХ ГОТОВЫ ДЛЯ МОДЕЛИРОВАНИЯ")
        print("=" * 60)
        return X, y, feature_cols

    def analyze_demand_trends(self) -> pd.DataFrame:
        if self.weekly_demand is None:
            raise ValueError("Сначала выполните aggregate_weekly_skills()")
            
        trends = self.weekly_demand.groupby('skill').agg({
            'vacancy_count': ['sum', 'mean'],
            'avg_salary': 'mean',
            'demand_level': lambda x: (x >= 3).sum() # Недели с Высоким/Очень высоким спросом
        }).reset_index()
        
        trends.columns = ['skill', 'total_vacancies', 'avg_weekly_vacancies', 
                          'avg_salary', 'high_demand_weeks']
        trends = trends.sort_values('total_vacancies', ascending=False)
        return trends
