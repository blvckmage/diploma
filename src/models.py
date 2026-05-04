#!/usr/bin/env python3
"""
Модуль ML моделей для прогнозирования востребованных профессий
Содержит классы для классификации, регрессии и оценки моделей
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import label_binarize

# Models
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# XGBoost and LightGBM
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class BaseModel:
    """Базовый класс для ML моделей"""
    
    def __init__(self, model_type: str = 'classification'):
        """
        Args:
            model_type: 'classification' или 'regression'
        """
        self.model_type = model_type
        self.model = None
        self.best_params = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Обучение модели"""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание"""
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей (только для классификации)"""
        if self.model_type != 'classification':
            raise ValueError("predict_proba доступен только для классификации")
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        return self.model.predict_proba(X)


class ProfessionClassifier(BaseModel):
    """Классификатор профессий"""
    
    # Доступные модели
    MODELS = {
        'logistic': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'knn': KNeighborsClassifier,
        'svm': SVC,
    }
    
    if XGBOOST_AVAILABLE:
        MODELS['xgboost'] = XGBClassifier
    if LIGHTGBM_AVAILABLE:
        MODELS['lightgbm'] = LGBMClassifier
    
    # Гиперпараметры по умолчанию
    DEFAULT_PARAMS = {
        'logistic': {'max_iter': 1000},
        'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1},
        'gradient_boosting': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
        'knn': {'n_neighbors': 5},
        'svm': {'probability': True, 'random_state': 42},
        'xgboost': {'n_estimators': 100, 'max_depth': 6, 'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'mlogloss'},
        'lightgbm': {'n_estimators': 100, 'max_depth': 6, 'random_state': 42, 'verbose': -1},
    }
    
    def __init__(self, model_name: str = 'random_forest', params: Dict = None):
        """
        Args:
            model_name: Название модели
            params: Гиперпараметры модели
        """
        super().__init__(model_type='classification')
        
        if model_name not in self.MODELS:
            raise ValueError(f"Неизвестная модель: {model_name}. Доступные: {list(self.MODELS.keys())}")
        
        self.model_name = model_name
        self.params = params or self.DEFAULT_PARAMS.get(model_name, {})
        self.model = self.MODELS[model_name](**self.params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ProfessionClassifier':
        """Обучение классификатора"""
        print(f"🔧 Обучение модели: {self.model_name}")
        self.model.fit(X, y)
        self.is_fitted = True
        print(f"✅ Модель обучена")
        return self
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, average: str = 'weighted') -> Dict[str, float]:
        """
        Оценка модели
        
        Args:
            X: Признаки
            y: Истинные метки
            average: Метод усреднения для метрик
        
        Returns:
            Словарь с метриками
        """
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average=average, zero_division=0),
            'recall': recall_score(y, y_pred, average=average, zero_division=0),
            'f1': f1_score(y, y_pred, average=average, zero_division=0),
        }
        
        return metrics
    
    def get_classification_report(self, X: np.ndarray, y: np.ndarray, target_names: List[str] = None) -> str:
        """Получение детального отчёта классификации"""
        y_pred = self.predict(X)
        return classification_report(y, y_pred, target_names=target_names)
    
    def get_confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Получение матрицы ошибок"""
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Кросс-валидация
        
        Args:
            X: Признаки
            y: Метки
            cv: Количество фолдов
        
        Returns:
            Словарь с результатами
        """
        print(f"🔄 Кросс-валидация ({cv} фолдов)...")
        
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = cross_val_score(self.model, X, y, cv=kfold, scoring='f1_weighted')
        
        results = {
            'mean_f1': scores.mean(),
            'std_f1': scores.std(),
            'all_scores': scores
        }
        
        print(f"   Mean F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return results
    
    def hyperparameter_search(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        param_grid: Dict = None,
        search_type: str = 'grid',
        cv: int = 3,
        n_iter: int = 10
    ) -> 'ProfessionClassifier':
        """
        Подбор гиперпараметров
        
        Args:
            X: Признаки
            y: Метки
            param_grid: Сетка параметров
            search_type: 'grid' или 'random'
            cv: Количество фолдов
            n_iter: Количество итераций для random search
        
        Returns:
            Self с лучшими параметрами
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        print(f"🔍 Подбор гиперпараметров ({search_type} search)...")
        
        if search_type == 'grid':
            search = GridSearchCV(
                self.model, param_grid, cv=cv, 
                scoring='f1_weighted', n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                self.model, param_grid, cv=cv, n_iter=n_iter,
                scoring='f1_weighted', n_jobs=-1, verbose=1, random_state=42
            )
        
        search.fit(X, y)
        
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        self.is_fitted = True
        
        print(f"✅ Лучшие параметры: {search.best_params_}")
        print(f"   Лучший F1: {search.best_score_:.4f}")
        
        return self
    
    def _get_default_param_grid(self) -> Dict:
        """Получение сетки параметров по умолчанию"""
        grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'logistic': {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs', 'saga']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, -1],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        return grids.get(self.model_name, {})


class ModelComparator:
    """Класс для сравнения нескольких моделей"""
    
    def __init__(self, models: Dict[str, BaseModel] = None):
        """
        Args:
            models: Словарь {имя: модель}
        """
        self.models = models or {}
        self.results = {}
    
    def add_model(self, name: str, model: BaseModel) -> None:
        """Добавление модели"""
        self.models[name] = model
    
    def compare(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray,
        cv: int = 5
    ) -> pd.DataFrame:
        """
        Сравнение всех моделей
        
        Args:
            X_train, y_train: Обучающая выборка
            X_test, y_test: Тестовая выборка
            cv: Количество фолдов для кросс-валидации
        
        Returns:
            DataFrame с результатами
        """
        print("=" * 60)
        print("📊 СРАВНЕНИЕ МОДЕЛЕЙ")
        print("=" * 60)
        
        results = []
        
        for name, model in self.models.items():
            print(f"\n🔧 Обучение: {name}")
            
            # Обучение
            model.fit(X_train, y_train)
            
            # Оценка
            train_metrics = model.evaluate(X_train, y_train)
            test_metrics = model.evaluate(X_test, y_test)
            
            # Кросс-валидация
            cv_results = model.cross_validate(X_train, y_train, cv=cv)
            
            results.append({
                'Model': name,
                'Train_Accuracy': train_metrics['accuracy'],
                'Test_Accuracy': test_metrics['accuracy'],
                'Train_F1': train_metrics['f1'],
                'Test_F1': test_metrics['f1'],
                'CV_Mean_F1': cv_results['mean_f1'],
                'CV_Std_F1': cv_results['std_f1']
            })
        
        self.results = pd.DataFrame(results).sort_values('Test_F1', ascending=False)
        
        print("\n" + "=" * 60)
        print("📋 РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
        print("=" * 60)
        print(self.results.to_string(index=False))
        
        return self.results
    
    def get_best_model(self, metric: str = 'Test_F1') -> Tuple[str, BaseModel]:
        """Получение лучшей модели по метрике"""
        if self.results.empty:
            raise ValueError("Сначала выполните compare()")
        
        best_name = self.results.sort_values(metric, ascending=False).iloc[0]['Model']
        return best_name, self.models[best_name]


class SalaryPredictor(BaseModel):
    """Регрессор для предсказания зарплаты"""
    
    MODELS = {
        'ridge': Ridge,
        'random_forest': RandomForestRegressor,
    }
    
    if XGBOOST_AVAILABLE:
        MODELS['xgboost'] = XGBRegressor
    if LIGHTGBM_AVAILABLE:
        MODELS['lightgbm'] = LGBMRegressor
    
    def __init__(self, model_name: str = 'random_forest', params: Dict = None):
        super().__init__(model_type='regression')
        
        if model_name not in self.MODELS:
            raise ValueError(f"Неизвестная модель: {model_name}")
        
        self.model_name = model_name
        self.params = params or {'random_state': 42}
        self.model = self.MODELS[model_name](**self.params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SalaryPredictor':
        """Обучение регрессора"""
        # Удаляем NaN из y
        mask = ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]
        
        print(f"🔧 Обучение регрессора: {self.model_name}")
        print(f"   Выборка: {len(y_clean)} записей (удалено {len(y) - len(y_clean)} без зарплаты)")
        
        self.model.fit(X_clean, y_clean)
        self.is_fitted = True
        print(f"✅ Регрессор обучен")
        
        return self
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Оценка регрессора"""
        mask = ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]
        
        y_pred = self.predict(X_clean)
        
        metrics = {
            'mae': mean_absolute_error(y_clean, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_clean, y_pred)),
            'r2': r2_score(y_clean, y_pred),
            'mape': np.mean(np.abs((y_clean - y_pred) / y_clean)) * 100
        }
        
        return metrics


class DemandRegressor(BaseModel):
    """Регрессор для предсказания спроса"""
    MODELS = {
        'ridge': Ridge,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor
    }
    
    if XGBOOST_AVAILABLE:
        MODELS['xgboost'] = XGBRegressor
    if LIGHTGBM_AVAILABLE:
        MODELS['lightgbm'] = LGBMRegressor
    
    def __init__(self, model_name: str = 'random_forest', params: Dict = None):
        super().__init__(model_type='regression')
        
        if model_name not in self.MODELS:
            raise ValueError(f"Неизвестная модель: {model_name}")
        
        self.model_name = model_name
        self.params = params or {'random_state': 42}
        self.model = self.MODELS[model_name](**self.params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DemandRegressor':
        print(f"🔧 Обучение регрессора: {self.model_name}")
        self.model.fit(X, y)
        self.is_fitted = True
        print(f"✅ Регрессор обучен")
        return self
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }
        return metrics

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        from sklearn.model_selection import KFold
        print(f"🔄 Кросс-валидация ({cv} фолдов)...")
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=kfold, scoring='r2')
        results = {
            'mean_r2': scores.mean(),
            'std_r2': scores.std(),
            'all_scores': scores
        }
        print(f"   Mean R2: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return results

class RegressorComparator:
    """Класс для сравнения регрессоров"""
    def __init__(self, models: Dict[str, BaseModel] = None):
        self.models = models or {}
        self.results = {}
    
    def add_model(self, name: str, model: BaseModel) -> None:
        self.models[name] = model
    
    def compare(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, cv: int = 5) -> pd.DataFrame:
        print("=" * 60)
        print("📊 СРАВНЕНИЕ РЕГРЕССОРОВ")
        print("=" * 60)
        results = []
        for name, model in self.models.items():
            print(f"\n🔧 Обучение: {name}")
            model.fit(X_train, y_train)
            train_metrics = model.evaluate(X_train, y_train)
            test_metrics = model.evaluate(X_test, y_test)
            cv_results = model.cross_validate(X_train, y_train, cv=cv)
            results.append({
                'Model': name,
                'Train_R2': train_metrics['r2'],
                'Test_R2': test_metrics['r2'],
                'Train_MAE': train_metrics['mae'],
                'Test_MAE': test_metrics['mae'],
                'Test_RMSE': test_metrics['rmse'],
                'CV_Mean_R2': cv_results['mean_r2']
            })
        self.results = pd.DataFrame(results).sort_values('Test_R2', ascending=False)
        print("\n" + "=" * 60)
        print("📋 РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
        print("=" * 60)
        print(self.results.to_string(index=False))
        return self.results
    
    def get_best_model(self, metric: str = 'Test_R2') -> Tuple[str, BaseModel]:
        if self.results.empty:
            raise ValueError("Сначала выполните compare()")
        best_name = self.results.sort_values(metric, ascending=False).iloc[0]['Model']
        return best_name, self.models[best_name]


def prepare_data_for_modeling(
    df: pd.DataFrame,
    target_column: str = 'Job',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Подготовка данных для моделирования
    
    Args:
        df: DataFrame с признаками и целевой переменной
        target_column: Название колонки с целевой переменной
        test_size: Доля тестовой выборки
        random_state: Seed для воспроизводимости
    
    Returns:
        X_train, X_test, y_train, y_test, feature_columns
    """
    # Определяем признаки
    feature_patterns = [
        'has_', 'tfidf_', '_encoded', '_scaled',
        'skills_count', 'text_length', 'word_count',
        'salary_by_', 'job_frequency', 'employer_vacancy_count',
        'experience_level', 'is_junior', 'is_senior', 'has_salary'
    ]
    
    feature_columns = []
    for col in df.columns:
        if any(pattern in col for pattern in feature_patterns):
            if col != 'target':
                feature_columns.append(col)
    
    X = df[feature_columns].values
    y = df[target_column].values if target_column in df.columns else df['target'].values
    
    # Заполняем пропуски
    X = np.nan_to_num(X, nan=0)
    
    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"📊 Данные подготовлены:")
    print(f"   • Обучающая выборка: {X_train.shape[0]} сэмплов")
    print(f"   • Тестовая выборка: {X_test.shape[0]} сэмплов")
    print(f"   • Признаков: {len(feature_columns)}")
    print(f"   • Классов: {len(np.unique(y))}")
    
    return X_train, X_test, y_train, y_test, feature_columns


if __name__ == "__main__":
    # Тестирование
    from preprocessing import VacancyPreprocessor
    from features import FeatureEngineer
    
    # Подготовка данных
    preprocessor = VacancyPreprocessor()
    df = preprocessor.load_data('results.csv')
    processed = preprocessor.preprocess()
    
    fe = FeatureEngineer()
    X, y, features = fe.prepare_for_modeling(processed, tfidf_max_features=20)
    
    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Сравнение моделей
    comparator = ModelComparator()
    comparator.add_model('Random Forest', ProfessionClassifier('random_forest'))
    comparator.add_model('Logistic', ProfessionClassifier('logistic'))
    
    if LIGHTGBM_AVAILABLE:
        comparator.add_model('LightGBM', ProfessionClassifier('lightgbm'))
    
    results = comparator.compare(X_train, y_train, X_test, y_test, cv=3)
    
    print("\n📊 Лучший результат:")
    best_name, best_model = comparator.get_best_model()
    print(f"   Модель: {best_name}")
    print(f"   Test F1: {results[results['Model'] == best_name]['Test_F1'].values[0]:.4f}")