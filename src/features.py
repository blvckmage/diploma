#!/usr/bin/env python3
"""
Модуль Feature Engineering для вакансий
Создание признаков для ML моделей
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
import re

# NLP imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD


class FeatureEngineer:
    """Класс для создания признаков из данных вакансий"""

    # Алиасы: синонимы → каноническое название
    SKILL_ALIASES = {
        'js':           'javascript',
        'ts':           'typescript',
        'go':           'golang',
        'k8s':          'kubernetes',
        'postgres':     'postgresql',
        'sklearn':      'scikit-learn',
        'node.js':      'nodejs',
        'node':         'nodejs',
        'nestjs':       'nodejs',
        'ml':           'machine learning',
        'deep learning':'machine learning',
        'ci/cd':        'cicd',
    }

    # Словарь IT навыков для извлечения (только канонические имена)
    IT_SKILLS = {
        # Языки программирования
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#',
        'golang', 'rust', 'kotlin', 'swift', 'php', 'ruby', 'scala',
        # Базы данных
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle',
        'sqlite', 'elasticsearch', 'cassandra', 'dynamodb',
        # Big Data
        'hadoop', 'spark', 'kafka', 'hive', 'airflow', 'dbt', 'snowflake',
        # ML/AI
        'machine learning', 'nlp', 'tensorflow', 'pytorch',
        'keras', 'scikit-learn', 'opencv', 'transformers',
        # Cloud
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'cicd', 'jenkins',
        # Frameworks
        'react', 'vue', 'angular', 'nodejs', 'django', 'flask', 'spring',
        'fastapi', 'express',
        # Tools
        'git', 'linux', 'bash', 'jira', 'confluence', 'figma', 'postman',
        # BI
        'tableau', 'power bi', 'excel', 'powerpoint'
    }

    # Русские IT навыки
    IT_SKILLS_RU = {
        'sql', 'питон', 'джава', 'скл', 'машинное обучение',
        'нейросети', 'анализ данных', 'big data', 'биг дата'
    }
    
    def __init__(self):
        self.df = None
        self.features_df = None
        self.label_encoders = {}
        self.scalers = {}
        self.vectorizers = {}
    
    def set_data(self, df: pd.DataFrame) -> None:
        """Установка DataFrame для обработки"""
        self.df = df.copy()
        print(f"✅ Данные установлены: {len(self.df)} записей")
    
    def normalize_skill(self, skill: str) -> str:
        """Нормализует навык через таблицу алиасов"""
        return self.SKILL_ALIASES.get(skill, skill)

    def extract_skills(self, text: str) -> List[str]:
        """
        Извлечение IT навыков из текста с нормализацией синонимов.
        """
        if not text or pd.isna(text):
            return []

        text_lower = text.lower()
        found_skills = set()

        # Расширенный словарь для поиска: канонические + алиасы
        search_terms = set(self.IT_SKILLS) | set(self.IT_SKILLS_RU) | set(self.SKILL_ALIASES.keys())

        for term in search_terms:
            escaped = re.escape(term)
            pattern = rf'(?<![a-z0-9а-я]){escaped}(?![a-z0-9а-я])'
            if re.search(pattern, text_lower):
                # Приводим к каноническому имени
                found_skills.add(self.normalize_skill(term))

        return list(found_skills)
    
    def create_skill_features(self, df: pd.DataFrame, text_column: str = 'clean_text') -> pd.DataFrame:
        """
        Создание бинарных признаков навыков
        
        Args:
            df: DataFrame с текстом
            text_column: Колонка с текстом
        
        Returns:
            DataFrame с новыми признаками
        """
        df = df.copy()
        
        print("🔧 Создание признаков навыков...")
        
        # Извлечение навыков
        df['skills_extracted'] = df[text_column].apply(self.extract_skills)
        df['skills_count'] = df['skills_extracted'].apply(len)
        
        # Бинарные признаки для топ навыков
        all_skills_found = []
        for skills in df['skills_extracted']:
            all_skills_found.extend(skills)
        
        top_skills = [s for s, _ in Counter(all_skills_found).most_common(30)]
        
        for skill in top_skills:
            col_name = f'has_{skill.replace(" ", "_").replace("/", "_")}'
            df[col_name] = df['skills_extracted'].apply(lambda x: 1 if skill in x else 0)
        
        print(f"   • Создано {len(top_skills)} бинарных признаков навыков")
        
        return df
    
    def create_tfidf_features(
        self, 
        df: pd.DataFrame, 
        text_column: str = 'clean_text',
        max_features: int = 100,
        ngram_range: Tuple[int, int] = (1, 2)
    ) -> pd.DataFrame:
        """
        Создание TF-IDF признаков из текста
        
        Args:
            df: DataFrame с текстом
            text_column: Колонка с текстом
            max_features: Максимальное количество признаков
            ngram_range: Диапазон n-грамм
        
        Returns:
            DataFrame с TF-IDF признаками
        """
        df = df.copy()
        
        print(f"🔧 Создание TF-IDF признаков (max_features={max_features})...")
        
        # TF-IDF Vectorizer (IT-навыки не исключаем — они и есть ключевые слова)
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=3,
            max_df=0.95,
        )
        
        # Обучение и трансформация
        tfidf_matrix = vectorizer.fit_transform(df[text_column].fillna(''))
        
        # Создание DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])],
            index=df.index
        )
        
        # Объединение
        df = pd.concat([df, tfidf_df], axis=1)
        
        # Сохранение vectorizer
        self.vectorizers['tfidf'] = vectorizer
        
        print(f"   • Создано {tfidf_matrix.shape[1]} TF-IDF признаков")
        
        return df
    
    def create_categorical_features(
        self, 
        df: pd.DataFrame,
        columns: List[str] = None,
        method: str = 'label'
    ) -> pd.DataFrame:
        """
        Кодирование категориальных признаков
        
        Args:
            df: DataFrame
            columns: Колонки для кодирования
            method: Метод кодирования ('label' или 'onehot')
        
        Returns:
            DataFrame с закодированными признаками
        """
        df = df.copy()
        
        if columns is None:
            columns = ['experience_clean', 'city_clean', 'employment', 'schedule']
        
        # Фильтруем только существующие колонки
        columns = [c for c in columns if c in df.columns]
        
        print(f"🔧 Кодирование категориальных признаков ({method})...")
        
        if method == 'label':
            for col in columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"   • {col}: {len(le.classes_)} категорий")
        
        elif method == 'onehot':
            for col in columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                print(f"   • {col}: {dummies.shape[1]} колонок")
        
        return df
    
    def create_numerical_features(
        self, 
        df: pd.DataFrame,
        columns: List[str] = None,
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Масштабирование числовых признаков
        
        Args:
            df: DataFrame
            columns: Колонки для масштабирования
            method: Метод масштабирования ('standard' или 'minmax')
        
        Returns:
            DataFrame с масштабированными признаками
        """
        df = df.copy()
        
        if columns is None:
            columns = ['salary_numeric', 'text_length', 'word_count', 'skills_count']
        
        # Фильтруем существующие колонки
        columns = [c for c in columns if c in df.columns]
        
        print(f"🔧 Масштабирование числовых признаков ({method})...")
        
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        for col in columns:
            # Заполняем пропуски медианой
            df[f'{col}_scaled'] = scaler.fit_transform(
                df[[col]].fillna(df[col].median())
            )
            self.scalers[col] = scaler
            print(f"   • {col}")
        
        return df
    
    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание агрегированных признаков по категориям
        
        Args:
            df: DataFrame
        
        Returns:
            DataFrame с агрегированными признаками
        """
        df = df.copy()
        
        print("🔧 Создание агрегированных признаков...")
        
        # Средняя зарплата по профессии
        if 'salary_numeric' in df.columns and 'Job' in df.columns:
            df['salary_by_job_mean'] = df.groupby('Job')['salary_numeric'].transform('mean')
            df['salary_by_job_std'] = df.groupby('Job')['salary_numeric'].transform('std')
            print("   • salary_by_job")
        
        # Средняя зарплата по городу
        if 'salary_numeric' in df.columns and 'city_clean' in df.columns:
            df['salary_by_city_mean'] = df.groupby('city_clean')['salary_numeric'].transform('mean')
            print("   • salary_by_city")
        
        # Количество вакансий по профессии
        if 'Job' in df.columns:
            df['job_vacancy_count'] = df.groupby('Job')['Job'].transform('count')
            df['job_frequency'] = df['job_vacancy_count'] / len(df)
            print("   • job_frequency")
        
        # Количество вакансий по работодателю
        if 'job' in df.columns:
            df['employer_vacancy_count'] = df.groupby('job')['job'].transform('count')
            print("   • employer_vacancy_count")
        
        return df
    
    def create_experience_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание признаков на основе опыта работы
        
        Args:
            df: DataFrame
        
        Returns:
            DataFrame с признаками опыта
        """
        df = df.copy()
        
        print("🔧 Создание признаков опыта...")
        
        # Числовое представление опыта
        exp_map = {
            'No experience': 0,
            '1-3 years': 1,
            '3-6 years': 2,
            '6+ years': 3,
            'Unknown': -1
        }
        
        if 'experience_clean' in df.columns:
            df['experience_level'] = df['experience_clean'].map(exp_map).fillna(-1)
            df['is_junior'] = (df['experience_level'] <= 1).astype(int)
            df['is_senior'] = (df['experience_level'] >= 2).astype(int)
            print("   • experience_level, is_junior, is_senior")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Получение списка колонок с признаками для ML
        
        Args:
            df: DataFrame
        
        Returns:
            Список колонок признаков
        """
        # Паттерны для поиска признаков
        patterns = [
            'has_', 'tfidf_', '_encoded', '_scaled', 
            'skills_count', 'text_length', 'word_count',
            'salary_by_', 'job_frequency', 'employer_vacancy_count',
            'experience_level', 'is_junior', 'is_senior', 'has_salary'
        ]
        
        feature_cols = []
        for col in df.columns:
            if any(pattern in col for pattern in patterns):
                feature_cols.append(col)
        
        return feature_cols
    
    def prepare_for_modeling(
        self,
        df: pd.DataFrame,
        target_column: str = 'Job',
        use_tfidf: bool = True,
        use_skills: bool = True,
        tfidf_max_features: int = 50
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Полный пайплайн подготовки данных для моделирования
        
        Args:
            df: DataFrame с предобработанными данными
            target_column: Целевая переменная
            use_tfidf: Использовать TF-IDF признаки
            use_skills: Использовать признаки навыков
            tfidf_max_features: Количество TF-IDF признаков
        
        Returns:
            Кортеж (X, y, feature_columns)
        """
        print("=" * 60)
        print("🚀 ПОЛНЫЙ ПАЙПЛАЙН FEATURE ENGINEERING")
        print("=" * 60)
        
        self.df = df.copy()
        
        # 1. Признаки навыков
        if use_skills:
            self.df = self.create_skill_features(self.df)
        
        # 2. TF-IDF признаки
        if use_tfidf:
            self.df = self.create_tfidf_features(
                self.df, 
                max_features=tfidf_max_features
            )
        
        # 3. Категориальные признаки
        self.df = self.create_categorical_features(self.df, method='label')
        
        # 4. Числовые признаки
        self.df = self.create_numerical_features(self.df)
        
        # 5. Агрегированные признаки
        self.df = self.create_aggregated_features(self.df)
        
        # 6. Признаки опыта
        self.df = self.create_experience_features(self.df)
        
        # Получение списка признаков
        feature_columns = self.get_feature_columns(self.df)
        
        # Целевая переменная
        if target_column in self.df.columns:
            y = self.df[target_column]
            # Кодирование целевой переменной
            le = LabelEncoder()
            y_encoded = le.fit_transform(y.astype(str))
            self.label_encoders['target'] = le
        else:
            y_encoded = None
        
        # Матрица признаков
        X = self.df[feature_columns].copy()
        
        # Заполнение пропусков
        X = X.fillna(0)
        
        print("\n" + "=" * 60)
        print(f"✅ FEATURE ENGINEERING ЗАВЕРШЁН")
        print(f"   • Признаков: {len(feature_columns)}")
        print(f"   • Записей: {len(X)}")
        if y_encoded is not None:
            print(f"   • Классов: {len(le.classes_)}")
        print("=" * 60)
        
        self.features_df = X
        
        return X, y_encoded, feature_columns
    
    def save_features(self, filepath: str) -> None:
        """Сохранение признаков в файл"""
        if self.features_df is not None:
            self.features_df.to_csv(filepath, index=False)
            print(f"✅ Признаки сохранены в {filepath}")


def get_top_skills_by_job(
    df: pd.DataFrame, 
    job_column: str = 'Job',
    text_column: str = 'clean_text',
    top_n: int = 10
) -> Dict[str, List[str]]:
    """
    Получение топ навыков для каждой профессии
    
    Args:
        df: DataFrame
        job_column: Колонка с профессией
        text_column: Колонка с текстом
        top_n: Количество топ навыков
    
    Returns:
        Словарь {профессия: [навыки]}
    """
    fe = FeatureEngineer()
    
    result = {}
    for job in df[job_column].unique():
        job_texts = df[df[job_column] == job][text_column].dropna()
        all_skills = []
        for text in job_texts:
            all_skills.extend(fe.extract_skills(text))
        
        top_skills = [s for s, _ in Counter(all_skills).most_common(top_n)]
        result[job] = top_skills
    
    return result


if __name__ == "__main__":
    # Тестирование
    from preprocessing import VacancyPreprocessor
    
    # Предобработка
    preprocessor = VacancyPreprocessor()
    df = preprocessor.load_data('results.csv')
    processed = preprocessor.preprocess()
    
    # Feature Engineering
    fe = FeatureEngineer()
    X, y, features = fe.prepare_for_modeling(processed)
    
    print(f"\n📊 Статистика признаков:")
    print(X.describe())
    
    # Топ навыки по профессиям
    print("\n🔧 Топ навыки по профессиям:")
    top_skills = get_top_skills_by_job(processed)
    for job, skills in list(top_skills.items())[:5]:
        print(f"   {job}: {', '.join(skills[:5])}")