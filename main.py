#!/usr/bin/env python3
"""
Главный скрипт для запуска полного пайплайна ML проекта
Usage: python main.py [--mode MODE] [--data PATH]
"""

import argparse
import sys
import os
from datetime import datetime

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import VacancyPreprocessor
from src.features import FeatureEngineer
from src.models import ProfessionClassifier, ModelComparator
from src.evaluation import ModelEvaluator, plot_class_distribution, plot_learning_curves

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def run_full_pipeline(data_path: str = 'results.csv', output_dir: str = 'reports/figures'):
    """
    Запуск полного пайплайна обучения модели
    
    Args:
        data_path: Путь к данным
        output_dir: Папка для сохранения результатов
    """
    print("=" * 70)
    print("🚀 ПОЛНЫЙ ПАЙПЛАЙН ML ПРОЕКТА")
    print("=" * 70)
    print(f"📅 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Создаём папки
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # ===== ЭТАП 1: ПРЕДОБРАБОТКА =====
    print("\n" + "=" * 50)
    print("📦 ЭТАП 1: ПРЕДОБРАБОТКА ДАННЫХ")
    print("=" * 50)
    
    preprocessor = VacancyPreprocessor()
    df = preprocessor.load_data(data_path)
    df_processed = preprocessor.preprocess()
    
    print(f"   Размер данных: {df_processed.shape}")
    
    # ===== ЭТАП 2: FEATURE ENGINEERING =====
    print("\n" + "=" * 50)
    print("🔧 ЭТАП 2: FEATURE ENGINEERING")
    print("=" * 50)
    
    fe = FeatureEngineer()
    X, y, feature_columns = fe.prepare_for_modeling(
        df_processed,
        target_column='Job',
        use_tfidf=True,
        use_skills=True,
        tfidf_max_features=50
    )
    
    print(f"   Признаков создано: {len(feature_columns)}")
    print(f"   Классов: {len(np.unique(y))}")
    
    # Сохраняем названия классов
    class_names = list(fe.label_encoders['target'].classes_)
    
    # ===== ЭТАП 3: РАЗДЕЛЕНИЕ ДАННЫХ =====
    print("\n" + "=" * 50)
    print("📊 ЭТАП 3: РАЗДЕЛЕНИЕ ДАННЫХ")
    print("=" * 50)
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    # ===== ЭТАП 4: СРАВНЕНИЕ МОДЕЛЕЙ =====
    print("\n" + "=" * 50)
    print("🤖 ЭТАП 4: СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 50)
    
    comparator = ModelComparator()
    comparator.add_model('Logistic Regression', ProfessionClassifier('logistic'))
    comparator.add_model('Random Forest', ProfessionClassifier('random_forest'))
    comparator.add_model('Gradient Boosting', ProfessionClassifier('gradient_boosting'))
    
    results = comparator.compare(X_train, y_train, X_test, y_test, cv=5)
    
    # ===== ЭТАП 5: ЛУЧШАЯ МОДЕЛЬ =====
    print("\n" + "=" * 50)
    print("🏆 ЭТАП 5: ВЫБОР И ОЦЕНКА ЛУЧШЕЙ МОДЕЛИ")
    print("=" * 50)
    
    best_name, best_model = comparator.get_best_model()
    print(f"   Лучшая модель: {best_name}")
    
    # Оценка
    evaluator = ModelEvaluator(
        best_model.model, X_train, X_test, y_train, y_test,
        feature_names=feature_columns, class_names=class_names
    )
    
    metrics = evaluator.get_metrics()
    
    print(f"\n   📈 Метрики на тестовой выборке:")
    print(f"      Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"      Precision: {metrics['test_precision']:.4f}")
    print(f"      Recall:    {metrics['test_recall']:.4f}")
    print(f"      F1-Score:  {metrics['test_f1']:.4f}")
    
    # ===== ЭТАП 6: ВИЗУАЛИЗАЦИИ =====
    print("\n" + "=" * 50)
    print("📊 ЭТАП 6: СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("=" * 50)
    
    # Confusion Matrix
    print("   Creating confusion matrix...")
    evaluator.plot_confusion_matrix(
        save_path=f'{output_dir}/confusion_matrix.png'
    )
    plt.close()
    
    # Classification Report
    print("   Creating classification report heatmap...")
    evaluator.plot_classification_report(
        save_path=f'{output_dir}/classification_report.png'
    )
    plt.close()
    
    # ROC Curves
    print("   Creating ROC curves...")
    evaluator.plot_roc_curves(
        save_path=f'{output_dir}/roc_curves.png'
    )
    plt.close()
    
    # Class Distribution
    print("   Creating class distribution plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    train_class_names = [class_names[i] for i in unique_train]
    axes[0].bar(range(len(unique_train)), counts_train, color='steelblue')
    axes[0].set_xticks(range(len(unique_train)))
    axes[0].set_xticklabels(train_class_names, rotation=45, ha='right')
    axes[0].set_title('Training Set Distribution', fontsize=14)
    axes[0].set_ylabel('Count')
    
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    test_class_names = [class_names[i] for i in unique_test]
    axes[1].bar(range(len(unique_test)), counts_test, color='coral')
    axes[1].set_xticks(range(len(unique_test)))
    axes[1].set_xticklabels(test_class_names, rotation=45, ha='right')
    axes[1].set_title('Test Set Distribution', fontsize=14)
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/class_distribution.png")
    
    # Model Comparison
    print("   Creating model comparison plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    results_sorted = results.sort_values('Test_F1', ascending=True)
    x = np.arange(len(results_sorted))
    width = 0.35
    
    ax.barh(x - width/2, results_sorted['Train_F1'], width, label='Train F1', color='lightcoral')
    ax.barh(x + width/2, results_sorted['Test_F1'], width, label='Test F1', color='steelblue')
    
    ax.set_yticks(x)
    ax.set_yticklabels(results_sorted['Model'])
    ax.set_xlabel('F1 Score')
    ax.set_title('Model Comparison: Train vs Test F1 Score', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/model_comparison.png")
    
    # ===== ЭТАП 7: СОХРАНЕНИЕ МОДЕЛИ =====
    print("\n" + "=" * 50)
    print("💾 ЭТАП 7: СОХРАНЕНИЕ МОДЕЛИ")
    print("=" * 50)
    
    import joblib
    
    joblib.dump(best_model.model, 'models/best_classifier.joblib')
    joblib.dump(fe.label_encoders, 'models/label_encoders.joblib')
    joblib.dump(feature_columns, 'models/feature_columns.joblib')
    
    print("   ✅ Model saved: models/best_classifier.joblib")
    print("   ✅ Encoders saved: models/label_encoders.joblib")
    print("   ✅ Features saved: models/feature_columns.joblib")
    
    # ===== ИТОГИ =====
    print("\n" + "=" * 70)
    print("✅ ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО!")
    print("=" * 70)
    
    print(f"\n📊 Результаты:")
    print(f"   • Лучшая модель: {best_name}")
    print(f"   • Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"   • Test F1-Score: {metrics['test_f1']:.4f}")
    
    print(f"\n📁 Сохранённые файлы:")
    print(f"   • {output_dir}/confusion_matrix.png")
    print(f"   • {output_dir}/classification_report.png")
    print(f"   • {output_dir}/roc_curves.png")
    print(f"   • {output_dir}/class_distribution.png")
    print(f"   • {output_dir}/model_comparison.png")
    print(f"   • models/best_classifier.joblib")
    
    print(f"\n📅 Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return best_model, metrics, results


def predict_single(title: str, skills: str, experience: str, city: str):
    """
    Предсказание профессии для одной вакансии
    
    Args:
        title: Название вакансии
        skills: Навыки (через запятую)
        experience: Опыт работы
        city: Город
    """
    import joblib
    
    # Загрузка модели
    model = joblib.load('models/best_classifier.joblib')
    encoders = joblib.load('models/label_encoders.joblib')
    
    # Подготовка признаков (упрощённо)
    # В реальном проекте нужен полный пайплайн
    
    print(f"\n🔮 Предсказание для вакансии:")
    print(f"   Title: {title}")
    print(f"   Skills: {skills}")
    print(f"   Experience: {experience}")
    print(f"   City: {city}")
    
    # Здесь должна быть подготовка признаков
    # prediction = model.predict(features)
    # profession = encoders['target'].inverse_transform(prediction)
    
    print("\n   ⚠️ Для предсказания нужен полный пайплайн предобработки")
    print("   Используйте notebooks/model_training.ipynb для предсказаний")


def main():
    parser = argparse.ArgumentParser(
        description='ML Pipeline для прогнозирования профессий'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        default='train',
        choices=['train', 'predict'],
        help='Режим работы: train (обучение) или predict (предсказание)'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='results.csv',
        help='Путь к файлу с данными'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='reports/figures',
        help='Папка для сохранения результатов'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_full_pipeline(data_path=args.data, output_dir=args.output)
    elif args.mode == 'predict':
        print("Режим предсказания")
        print("Используйте: python -c \"from src import *\" ")


if __name__ == "__main__":
    main()