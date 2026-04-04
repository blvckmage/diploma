"""
Src module for Predictive Analytics Model
"""

from .preprocessing import VacancyPreprocessor, remove_outliers_iqr, normalize_salary_by_city
from .features import FeatureEngineer, get_top_skills_by_job
from .models import ProfessionClassifier, ModelComparator, SalaryPredictor
from .evaluation import ModelEvaluator, SHAPAnalyzer, plot_class_distribution, plot_learning_curves

__all__ = [
    'VacancyPreprocessor',
    'FeatureEngineer',
    'ProfessionClassifier',
    'ModelComparator',
    'SalaryPredictor',
    'ModelEvaluator',
    'SHAPAnalyzer',
    'remove_outliers_iqr',
    'normalize_salary_by_city',
    'get_top_skills_by_job',
    'plot_class_distribution',
    'plot_learning_curves'
]
