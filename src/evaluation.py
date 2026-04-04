#!/usr/bin/env python3
"""
Модуль оценки и интерпретации ML моделей
Включает SHAP анализ, визуализации и метрики
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# SHAP for model interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ModelEvaluator:
    """Класс для комплексной оценки моделей"""
    
    def __init__(self, model, X_train: np.ndarray, X_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray,
                 feature_names: List[str] = None,
                 class_names: List[str] = None):
        """
        Args:
            model: Обученная модель
            X_train, X_test: Признаки
            y_train, y_test: Метки
            feature_names: Названия признаков
            class_names: Названия классов
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.class_names = class_names
        
        # Предсказания
        self.y_pred_train = model.predict(X_train)
        self.y_pred_test = model.predict(X_test)
        
        # Вероятности (если доступно)
        self.y_proba_test = None
        if hasattr(model, 'predict_proba'):
            self.y_proba_test = model.predict_proba(X_test)
    
    def get_metrics(self) -> Dict[str, float]:
        """Получение основных метрик"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'train_accuracy': accuracy_score(self.y_train, self.y_pred_train),
            'test_accuracy': accuracy_score(self.y_test, self.y_pred_test),
            'test_precision': precision_score(self.y_test, self.y_pred_test, average='weighted', zero_division=0),
            'test_recall': recall_score(self.y_test, self.y_pred_test, average='weighted', zero_division=0),
            'test_f1': f1_score(self.y_test, self.y_pred_test, average='weighted', zero_division=0),
        }
        
        return metrics
    
    def plot_confusion_matrix(self, figsize: Tuple[int, int] = (12, 10), 
                              save_path: str = None) -> None:
        """Визуализация матрицы ошибок"""
        cm = confusion_matrix(self.y_test, self.y_pred_test)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Сохранено: {save_path}")
        
        plt.show()
    
    def plot_classification_report(self, save_path: str = None) -> None:
        """Визуализация отчета классификации"""
        # Получаем уникальные классы из данных
        unique_classes = np.unique(np.concatenate([self.y_test, self.y_pred_test]))
        class_names_used = [self.class_names[i] for i in unique_classes] if self.class_names else None
        
        report = classification_report(self.y_test, self.y_pred_test,
                                        labels=unique_classes,
                                        target_names=class_names_used,
                                        output_dict=True)
        
        # Создаем DataFrame
        df_report = pd.DataFrame(report).transpose()
        df_report = df_report.iloc[:-3]  # Убираем итоговые строки
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(df_report[['precision', 'recall', 'f1-score']], 
                    annot=True, cmap='RdYlGn', fmt='.2f', ax=ax)
        ax.set_title('Classification Report by Class', fontsize=14)
        ax.set_ylabel('Class')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Сохранено: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = None) -> None:
        """Визуализация важности признаков"""
        if not hasattr(self.model, 'feature_importances_'):
            print("⚠️ Feature importance недоступна для этой модели")
            return
        
        importance = self.model.feature_importances_
        
        feat_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='viridis')
        plt.title(f'Top-{top_n} Feature Importances', fontsize=14)
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Сохранено: {save_path}")
        
        plt.show()
        
        return feat_imp
    
    def plot_roc_curves(self, save_path: str = None) -> None:
        """Построение ROC кривых для мультиклассовой классификации"""
        if self.y_proba_test is None:
            print("⚠️ Вероятности недоступны")
            return
        
        # Получаем уникальные классы
        unique_classes = np.unique(self.y_test)
        n_classes = len(unique_classes)
        n_proba_classes = self.y_proba_test.shape[1]
        
        # Бинаризация меток с правильными классами
        y_test_bin = label_binarize(self.y_test, classes=range(n_proba_classes))
        
        # Проверяем размерности
        if y_test_bin.shape[1] != n_proba_classes:
            # Если размерности не совпадают, упрощаем визуализацию
            print("   ⚠️ Упрощенная визуализация ROC")
            
            # Вычисляем только микро-усредненный ROC
            from sklearn.metrics import roc_auc_score
            try:
                auc_score = roc_auc_score(self.y_test, self.y_proba_test, multi_class='ovr')
                
                plt.figure(figsize=(10, 6))
                plt.bar(['Overall AUC'], [auc_score], color='steelblue')
                plt.ylabel('AUC Score')
                plt.title(f'Overall ROC AUC: {auc_score:.4f}', fontsize=14)
                plt.ylim([0, 1])
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"✅ Сохранено: {save_path}")
                
                plt.show()
            except Exception as e:
                print(f"   ⚠️ Невозможно построить ROC: {e}")
            return
        
        # ROC для каждого класса
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_proba_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], self.y_proba_test[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Микро-усреднение
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), self.y_proba_test.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Визуализация
        plt.figure(figsize=(12, 10))
        
        # Топ-5 классов по AUC
        top_classes = sorted(range(n_proba_classes), key=lambda i: roc_auc[i], reverse=True)[:5]
        
        for i in top_classes:
            class_name = self.class_names[i] if self.class_names and i < len(self.class_names) else f'Class {i}'
            plt.plot(fpr[i], tpr[i], lw=2,
                     label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
                 color='navy', linestyle=':', linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (Top-5 Classes)', fontsize=14)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Сохранено: {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """Генерация текстового отчета"""
        metrics = self.get_metrics()
        
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        report.append("\n📊 Metrics:")
        for name, value in metrics.items():
            report.append(f"   • {name}: {value:.4f}")
        
        report.append("\n📋 Classification Report:")
        report.append(classification_report(self.y_test, self.y_pred_test,
                                            target_names=self.class_names))
        
        return "\n".join(report)


class SHAPAnalyzer:
    """Класс для SHAP анализа моделей"""
    
    def __init__(self, model, X_train: np.ndarray, feature_names: List[str] = None):
        """
        Args:
            model: Обученная модель
            X_train: Обучающая выборка
            feature_names: Названия признаков
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP не установлен. Установите: pip install shap")
        
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.explainer = None
        self.shap_values = None
    
    def fit(self, sample_size: int = 100) -> None:
        """
        Вычисление SHAP значений
        
        Args:
            sample_size: Размер выборки для быстрого вычисления
        """
        print("🔄 Вычисление SHAP значений...")
        
        # Выборка для быстрого вычисления
        if len(self.X_train) > sample_size:
            indices = np.random.choice(len(self.X_train), sample_size, replace=False)
            X_sample = self.X_train[indices]
        else:
            X_sample = self.X_train
        
        # Создание explainer
        if hasattr(self.model, 'predict_proba'):
            self.explainer = shap.Explainer(self.model.predict_proba, X_sample)
        else:
            self.explainer = shap.Explainer(self.model, X_sample)
        
        self.shap_values = self.explainer(X_sample)
        print("✅ SHAP значения вычислены")
    
    def plot_summary(self, max_display: int = 20, save_path: str = None) -> None:
        """График важности признаков по SHAP"""
        if self.shap_values is None:
            self.fit()
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, features=self.X_train[:len(self.shap_values)],
                          feature_names=self.feature_names, max_display=max_display,
                          show=False)
        plt.title('SHAP Feature Importance', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Сохранено: {save_path}")
        
        plt.show()
    
    def plot_beeswarm(self, max_display: int = 20, save_path: str = None) -> None:
        """Beeswarm plot для SHAP значений"""
        if self.shap_values is None:
            self.fit()
        
        plt.figure(figsize=(12, 10))
        shap.plots.beeswarm(self.shap_values, max_display=max_display, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Сохранено: {save_path}")
        
        plt.show()
    
    def plot_waterfall(self, sample_idx: int = 0, save_path: str = None) -> None:
        """Waterfall plot для конкретного предсказания"""
        if self.shap_values is None:
            self.fit()
        
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(self.shap_values[sample_idx], show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Сохранено: {save_path}")
        
        plt.show()


def plot_class_distribution(y: np.ndarray, class_names: List[str] = None,
                           title: str = "Class Distribution",
                           save_path: str = None) -> None:
    """Визуализация распределения классов"""
    unique, counts = np.unique(y, return_counts=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(unique)), counts, color='steelblue')
    plt.xticks(range(len(unique)), class_names, rotation=45, ha='right')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title, fontsize=14)
    
    # Добавляем значения на столбцы
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Сохранено: {save_path}")
    
    plt.show()


def plot_learning_curves(model, X: np.ndarray, y: np.ndarray,
                        cv: int = 5, save_path: str = None) -> None:
    """Построение кривых обучения"""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1_weighted'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Training F1')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='steelblue')
    plt.plot(train_sizes, test_mean, 'o-', color='coral', label='Validation F1')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                     alpha=0.1, color='coral')
    
    plt.xlabel('Training Size')
    plt.ylabel('F1 Score')
    plt.title('Learning Curves', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Сохранено: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    from preprocessing import VacancyPreprocessor
    from features import FeatureEngineer
    from models import ProfessionClassifier
    from sklearn.model_selection import train_test_split
    
    # Подготовка данных
    preprocessor = VacancyPreprocessor()
    df = preprocessor.load_data('results.csv')
    processed = preprocessor.preprocess()
    
    fe = FeatureEngineer()
    X, y, features = fe.prepare_for_modeling(processed, tfidf_max_features=20)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                         random_state=42, stratify=y)
    
    # Обучение модели
    model = ProfessionClassifier('random_forest')
    model.fit(X_train, y_train)
    
    # Оценка
    class_names = [fe.label_encoders['target'].classes_[i] 
                   for i in range(len(fe.label_encoders['target'].classes_))]
    
    evaluator = ModelEvaluator(model.model, X_train, X_test, y_train, y_test,
                               feature_names=features, class_names=class_names)
    
    print(evaluator.generate_report())
    
    # Визуализации
    evaluator.plot_confusion_matrix()
    evaluator.plot_feature_importance()
    
    # SHAP анализ
    if SHAP_AVAILABLE:
        shap_analyzer = SHAPAnalyzer(model.model, X_train, features)
        shap_analyzer.plot_summary()