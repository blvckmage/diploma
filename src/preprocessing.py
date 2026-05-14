#!/usr/bin/env python3
"""
Модуль предобработки данных о вакансиях
Содержит функции для очистки и подготовки данных
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Optional, List, Dict, Union


class VacancyPreprocessor:
    """Класс для предобработки данных о вакансиях"""
    
    # Маппинг опыта работы
    EXPERIENCE_MAP = {
        'Нет опыта': 'No experience',
        'От 1 года до 3 лет': '1-3 years',
        'От 3 до 6 лет': '3-6 years',
        'Более 6 лет': '6+ years'
    }
    
    # Русские стоп-слова
    STOPWORDS_RU = {
        'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'не', 'что', 'как', 'это', 
        'за', 'из', 'у', 'к', 'о', 'но', 'или', 'при', 'работы', 'работа', 
        'опыт', 'знание', 'умение', 'навыки', 'требования', 'обязанности', 
        'высшее', 'образование', 'команда', 'проект', 'задач', 'задачи', 
        'работать', 'поиск', 'подбор', 'кандидатов', 'поддержка', 'знания',
        'сфере', 'новых', 'компании', 'понимание', 'участие', 'менее', 
        'года', 'лет', 'год', 'месяц', 'х', '1', '2', '3', '4', '5', '6',
        'experience', 'in', 'years', 'of', 'with', 'and', 'for', 'the', 
        'to', 'a', 'an', 'is', 'are', 'be', 'will'
    }
    
    def __init__(self):
        self.df = None
        self.processed_df = None
    
    def load_data(self, filepath: str, encoding: str = 'utf-8-sig') -> pd.DataFrame:
        """
        Загрузка данных из CSV файла
        
        Args:
            filepath: Путь к файлу
            encoding: Кодировка файла
        
        Returns:
            DataFrame с данными
        """
        self.df = pd.read_csv(filepath, encoding=encoding)
        print(f"✅ Загружено {len(self.df)} вакансий")
        return self.df
    
    def parse_salary(self, salary_str: Optional[str]) -> float:
        """
        Парсинг зарплаты из строки в число
        
        Args:
            salary_str: Строка с зарплатой
        
        Returns:
            Числовое значение зарплаты (среднее если диапазон)
        """
        if pd.isna(salary_str) or str(salary_str).strip() == '':
            return np.nan
        
        s = str(salary_str).strip()
        numbers = re.findall(r'\d+', s)
        
        if not numbers:
            return np.nan
        
        nums = [int(n) for n in numbers]
        return np.mean(nums)
    
    def clean_html(self, text: Optional[str]) -> str:
        """
        Очистка HTML тегов из текста
        
        Args:
            text: Исходный текст
        
        Returns:
            Очищенный текст
        """
        if not text or pd.isna(text):
            return ""
        
        # Удаляем HTML теги
        clean = re.sub(r'<[^>]+>', '', str(text))
        # Удаляем множественные пробелы
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()
    
    def clean_text_for_nlp(self, text: Optional[str]) -> str:
        """
        Очистка текста для NLP обработки
        
        Args:
            text: Исходный текст
        
        Returns:
            Очищенный текст в нижнем регистре
        """
        if not text or pd.isna(text):
            return ""
        
        # Удаляем HTML теги
        text = re.sub(r'<[^>]+>', '', str(text))
        # Оставляем буквы, цифры и специальные символы для IT (+#)
        text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s\+\#\.]', ' ', text)
        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def parse_date(self, date_str: Optional[str], format: str = '%d-%m-%Y') -> Optional[datetime]:
        """
        Парсинг даты из строки
        
        Args:
            date_str: Строка с датой
            format: Формат даты
        
        Returns:
            Объект datetime или None
        """
        if pd.isna(date_str) or not date_str:
            return None
        
        try:
            return pd.to_datetime(date_str, format=format, errors='coerce')
        except:
            return None
    
    def preprocess(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Полный пайплайн предобработки данных
        
        Args:
            df: DataFrame для обработки (если None, используется загруженный)
        
        Returns:
            Обработанный DataFrame
        """
        if df is not None:
            self.df = df
        
        if self.df is None:
            raise ValueError("Данные не загружены. Вызовите load_data() или передайте df")
        
        print("🔄 Начинаем предобработку данных...")
        
        # Создаем копию
        self.processed_df = self.df.copy()
        
        # 1. Парсинг даты
        print("   • Парсинг дат...")
        self.processed_df['publish_date'] = pd.to_datetime(
            self.processed_df['publish_date'], 
            format='%d-%m-%Y', 
            errors='coerce'
        )
        
        # 2. Парсинг зарплаты
        print("   • Парсинг зарплат...")
        self.processed_df['salary_numeric'] = self.processed_df['salary'].apply(self.parse_salary)
        self.processed_df = filter_salary_outliers(self.processed_df)

        # 3. Очистка опыта работы
        print("   • Очистка опыта работы...")
        self.processed_df['experience_clean'] = (
            self.processed_df['experience']
            .map(self.EXPERIENCE_MAP)
            .fillna('Unknown')
        )
        
        # 4. Очистка HTML из текстовых полей
        print("   • Очистка HTML...")
        self.processed_df['requirements_clean'] = self.processed_df['requirements'].apply(self.clean_html)
        self.processed_df['responsibilities_clean'] = self.processed_df['responsibilities'].apply(self.clean_html)
        
        # 5. Создание объединенного текста для NLP
        print("   • Создание объединенного текста...")
        self.processed_df['full_text'] = (
            self.processed_df['requirements_clean'].fillna('') + ' ' + 
            self.processed_df['responsibilities_clean'].fillna('')
        )
        
        # 6. Очистка текста для NLP
        print("   • Очистка текста для NLP...")
        self.processed_df['clean_text'] = self.processed_df['full_text'].apply(self.clean_text_for_nlp)
        
        # 7. Временные признаки
        print("   • Извлечение временных признаков...")
        self.processed_df['day_of_week'] = self.processed_df['publish_date'].dt.day_name()
        self.processed_df['day_of_month'] = self.processed_df['publish_date'].dt.day
        self.processed_df['month'] = self.processed_df['publish_date'].dt.month
        self.processed_df['week_of_year'] = self.processed_df['publish_date'].dt.isocalendar().week
        
        # 8. Длина текста
        print("   • Вычисление длины текста...")
        self.processed_df['text_length'] = self.processed_df['clean_text'].str.len()
        self.processed_df['word_count'] = self.processed_df['clean_text'].str.split().str.len()
        
        # 9. Флаг наличия зарплаты
        self.processed_df['has_salary'] = self.processed_df['salary_numeric'].notna().astype(int)
        
        # 10. Очистка названий городов
        self.processed_df['city_clean'] = self.processed_df['city'].str.strip()
        
        print(f"✅ Предобработка завершена! Размер: {self.processed_df.shape}")
        
        return self.processed_df
    
    def get_missing_stats(self) -> pd.DataFrame:
        """
        Получение статистики по пропущенным значениям
        
        Returns:
            DataFrame со статистикой пропусков
        """
        if self.processed_df is None:
            raise ValueError("Данные не обработаны. Вызовите preprocess()")
        
        missing = self.processed_df.isnull().sum()
        pct = (missing / len(self.processed_df) * 100).round(2)
        
        return pd.DataFrame({
            'Missing Values': missing,
            'Percentage (%)': pct
        }).sort_values('Missing Values', ascending=False)
    
    def save_processed(self, filepath: str, encoding: str = 'utf-8-sig') -> None:
        """
        Сохранение обработанных данных
        
        Args:
            filepath: Путь для сохранения
            encoding: Кодировка файла
        """
        if self.processed_df is None:
            raise ValueError("Данные не обработаны. Вызовите preprocess()")
        
        self.processed_df.to_csv(filepath, index=False, encoding=encoding)
        print(f"✅ Данные сохранены в {filepath}")


def filter_salary_outliers(df: pd.DataFrame,
                           min_salary: float = 50_000,
                           max_salary: float = 10_000_000) -> pd.DataFrame:
    """
    Обнуляет salary_numeric для явно некорректных значений.
    Порог min_salary=50_000 KZT отсекает артефакты вроде 1 000 KZT.
    Порог max_salary=10_000_000 KZT отсекает случайные числа из текста.
    """
    df = df.copy()
    if 'salary_numeric' not in df.columns:
        return df

    bad_mask = (
        (df['salary_numeric'] < min_salary) |
        (df['salary_numeric'] > max_salary)
    )
    n_bad = bad_mask.sum()
    df.loc[bad_mask, 'salary_numeric'] = np.nan
    if n_bad:
        print(f"   ⚠️ Обнулено {n_bad} аномальных зарплат (< {min_salary:,} или > {max_salary:,} KZT)")
    return df


def remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Удаление выбросов методом IQR
    
    Args:
        df: DataFrame
        column: Колонка для анализа
        multiplier: Множитель для IQR
    
    Returns:
        DataFrame без выбросов
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def normalize_salary_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализация зарплаты относительно города
    
    Args:
        df: DataFrame с колонками salary_numeric и city
    
    Returns:
        DataFrame с нормализованной зарплатой
    """
    df = df.copy()
    city_means = df.groupby('city')['salary_numeric'].transform('mean')
    city_stds = df.groupby('city')['salary_numeric'].transform('std')
    
    df['salary_normalized'] = (df['salary_numeric'] - city_means) / city_stds
    
    return df


if __name__ == "__main__":
    # Тестирование модуля
    preprocessor = VacancyPreprocessor()
    
    # Загрузка и обработка
    df = preprocessor.load_data('results.csv')
    processed = preprocessor.preprocess()
    
    # Статистика пропусков
    print("\n=== Пропущенные значения ===")
    print(preprocessor.get_missing_stats())
    
    # Сохранение
    preprocessor.save_processed('data/processed/processed_vacancies.csv')