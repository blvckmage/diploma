#!/usr/bin/env python3
"""
Главный скрипт для прогнозирования востребованности профессий
Соответствует теме: "Development of a predictive analytics model for predicting sought-after professions"

Usage: python main_demand.py [--mode train]
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.demand_analysis import DemandAnalyzer
from src.models import ProfessionClassifier, ModelComparator
from src.evaluation import ModelEvaluator

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def run_demand_pipeline(data_path: str = 'results.csv', output_dir: str = 'reports/figures'):
    """
    Запуск пайплайна прогнозирования востребованности профессий
    
    Целевая переменная: demand_level (0=низкий, 1=средний, 2=высокий спрос)
    """
    print("=" * 70)
    print("🚀 ПРОГНОЗИРОВАНИЕ ВОСТРЕБОВАННОСТИ ПРОФЕССИЙ")
    print("=" * 70)
    print(f"📅 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📖 Тема: Predicting sought-after professions based on vacancy data")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # ===== ЭТАП 1: АНАЛИЗ ВОСТРЕБОВАННОСТИ =====
    print("\n" + "=" * 50)
    print("📊 ЭТАП 1: АНАЛИЗ ВОСТРЕБОВАННОСТИ ПРОФЕССИЙ")
    print("=" * 50)
    
    analyzer = DemandAnalyzer()
    analyzer.load_data(data_path)
    
    # Используем 5 классов для более сложной классификации
    X, y, feature_cols = analyzer.prepare_for_modeling(n_classes=5)
    
    # Показываем компоненты индекса
    print("\n📊 Компоненты индекса востребованности:")
    components = analyzer.get_demand_components_breakdown()
    print(components.head(10).to_string())
    
    # Анализ трендов
    print("\n📈 Анализ востребованности:")
    trends = analyzer.analyze_demand_trends()
    print(trends.head(10).to_string())
    
    # ===== ЭТАП 2: РАЗДЕЛЕНИЕ ДАННЫХ =====
    print("\n" + "=" * 50)
    print("📊 ЭТАП 2: РАЗДЕЛЕНИЕ ДАННЫХ")
    print("=" * 50)
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    # ===== ЭТАП 3: СРАВНЕНИЕ МОДЕЛЕЙ =====
    print("\n" + "=" * 50)
    print("🤖 ЭТАП 3: СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 50)
    
    comparator = ModelComparator()
    comparator.add_model('Logistic Regression', ProfessionClassifier('logistic'))
    comparator.add_model('Random Forest', ProfessionClassifier('random_forest'))
    comparator.add_model('Gradient Boosting', ProfessionClassifier('gradient_boosting'))
    
    results = comparator.compare(X_train.values, y_train.values, X_test.values, y_test.values, cv=5)
    
    # ===== ЭТАП 4: ЛУЧШАЯ МОДЕЛЬ =====
    print("\n" + "=" * 50)
    print("🏆 ЭТАП 4: ОЦЕНКА ЛУЧШЕЙ МОДЕЛИ")
    print("=" * 50)
    
    best_name, best_model = comparator.get_best_model()
    print(f"   Лучшая модель: {best_name}")
    
    # Оценка
    class_names = ['Очень низкий', 'Низкий', 'Средний', 'Высокий', 'Очень высокий']
    
    evaluator = ModelEvaluator(
        best_model.model, X_train.values, X_test.values, y_train.values, y_test.values,
        feature_names=feature_cols, class_names=class_names
    )
    
    metrics = evaluator.get_metrics()
    
    print(f"\n   📈 Метрики прогнозирования востребованности:")
    print(f"      Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"      Precision: {metrics['test_precision']:.4f}")
    print(f"      Recall:    {metrics['test_recall']:.4f}")
    print(f"      F1-Score:  {metrics['test_f1']:.4f}")
    
    # ===== ЭТАП 5: ВИЗУАЛИЗАЦИИ =====
    print("\n" + "=" * 50)
    print("📊 ЭТАП 5: СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("=" * 50)
    
    # Confusion Matrix
    print("   Creating confusion matrix...")
    evaluator.plot_confusion_matrix(save_path=f'{output_dir}/demand_confusion_matrix.png')
    plt.close()
    
    # Classification Report
    print("   Creating classification report...")
    evaluator.plot_classification_report(save_path=f'{output_dir}/demand_classification_report.png')
    plt.close()
    
    # Feature Importance
    print("   Creating feature importance plot...")
    evaluator.plot_feature_importance(save_path=f'{output_dir}/demand_feature_importance.png')
    plt.close()
    
    # Demand Distribution
    print("   Creating demand distribution plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Распределение классов (5 классов)
    demand_counts = y.value_counts().sort_index()
    colors = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
    class_labels = ['Очень\nнизкий', 'Низкий', 'Средний', 'Высокий', 'Очень\nвысокий']
    axes[0].bar(class_labels, 
                [demand_counts.get(i, 0) for i in range(5)],
                color=colors)
    axes[0].set_title('Распределение уровней востребованности', fontsize=12)
    axes[0].set_ylabel('Количество записей')
    
    # Топ профессии
    top_professions = trends.head(10)
    axes[1].barh(top_professions['profession'], top_professions['total_vacancies'], color='steelblue')
    axes[1].set_title('Топ-10 востребованных профессий', fontsize=12)
    axes[1].set_xlabel('Количество вакансий')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/demand_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/demand_distribution.png")
    
    # Model Comparison
    print("   Creating model comparison plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    results_sorted = results.sort_values('Test_F1', ascending=True)
    x = np.arange(len(results_sorted))
    width = 0.35
    
    ax.barh(x - width/2, results_sorted['Train_F1'], width, label='Train F1', color='lightcoral')
    ax.barh(x + width/2, results_sorted['Test_F1'], width, label='Test F1', color='steelblue')
    
    ax.set_yticks(x)
    ax.set_yticklabels(results_sorted['Model'])
    ax.set_xlabel('F1 Score')
    ax.set_title('Сравнение моделей прогнозирования востребованности', fontsize=12)
    ax.legend()
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/demand_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/demand_model_comparison.png")
    
    # ===== ЭТАП 6: СОХРАНЕНИЕ =====
    print("\n" + "=" * 50)
    print("💾 ЭТАП 6: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 50)
    
    import joblib
    
    joblib.dump(best_model.model, 'models/demand_classifier.joblib')
    joblib.dump(feature_cols, 'models/demand_features.joblib')
    
    print("   ✅ Model saved: models/demand_classifier.joblib")
    print("   ✅ Features saved: models/demand_features.joblib")
    
    # Сохранение анализа востребованности
    trends.to_csv('reports/demand_analysis.csv', index=False)
    print("   ✅ Analysis saved: reports/demand_analysis.csv")
    
    # ===== ИТОГИ =====
    print("\n" + "=" * 70)
    print("✅ ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО!")
    print("=" * 70)
    
    print(f"\n📊 Результаты прогнозирования востребованности:")
    print(f"   • Лучшая модель: {best_name}")
    print(f"   • Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"   • Test F1-Score: {metrics['test_f1']:.4f}")
    
    print(f"\n🏆 Самые востребованные профессии:")
    for i, prof in enumerate(analyzer.get_most_sought_after_professions(5), 1):
        print(f"   {i}. {prof}")
    
    print(f"\n📁 Сохранённые файлы:")
    print(f"   • {output_dir}/demand_confusion_matrix.png")
    print(f"   • {output_dir}/demand_classification_report.png")
    print(f"   • {output_dir}/demand_feature_importance.png")
    print(f"   • {output_dir}/demand_distribution.png")
    print(f"   • {output_dir}/demand_model_comparison.png")
    print(f"   • models/demand_classifier.joblib")
    print(f"   • reports/demand_analysis.csv")
    
    print(f"\n📅 Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return best_model, metrics, trends


def predict_demand(data_path: str = 'results.csv'):
    """
    Прогнозирование востребованности с использованием обученной модели
    
    Usage: python main_demand.py --mode predict
    """
    import joblib
    
    print("=" * 70)
    print("🔮 ПРОГНОЗИРОВАНИЕ ВОСТРЕБОВАННОСТИ ПРОФЕССИЙ")
    print("=" * 70)
    
    # Загрузка модели
    model_path = 'models/demand_classifier.joblib'
    features_path = 'models/demand_features.joblib'
    
    if not os.path.exists(model_path):
        print("❌ Модель не найдена! Сначала запустите обучение:")
        print("   python main_demand.py --mode train")
        return
    
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    print(f"✅ Модель загружена: {model_path}")
    
    # Подготовка данных
    analyzer = DemandAnalyzer()
    analyzer.load_data(data_path)
    analyzer.aggregate_weekly_demand()
    analyzer.calculate_demand_index()
    analyzer.create_demand_target(n_classes=5)
    analyzer.create_features_for_demand()
    
    X = analyzer.weekly_demand[feature_cols]
    
    # Прогнозирование
    predictions = model.predict(X.values)
    probabilities = model.predict_proba(X.values)
    
    # Добавление результатов
    analyzer.weekly_demand['predicted_demand_level'] = predictions
    
    # Названия классов
    class_names = {0: 'Очень низкий', 1: 'Низкий', 2: 'Средний', 3: 'Высокий', 4: 'Очень высокий'}
    analyzer.weekly_demand['predicted_demand_name'] = [class_names[p] for p in predictions]
    
    # Результаты
    print("\n" + "=" * 70)
    print("📊 РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ")
    print("=" * 70)
    
    results_df = analyzer.weekly_demand[['profession', 'vacancy_count', 'demand_index', 
                                          'demand_level', 'predicted_demand_level', 
                                          'predicted_demand_name']].copy()
    
    # Добавляем вероятности
    results_df['confidence'] = [f"{prob.max():.2%}" for prob in probabilities]
    
    print(results_df.to_string())
    
    # Точность
    actual = analyzer.weekly_demand['demand_level'].values
    accuracy = (predictions == actual).mean()
    print(f"\n📈 Accuracy: {accuracy:.2%}")
    
    # Сохранение результатов
    results_df.to_csv('reports/demand_predictions.csv', index=False)
    print(f"\n✅ Результаты сохранены: reports/demand_predictions.csv")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Прогнозирование востребованности профессий'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        default='train',
        choices=['train', 'predict'],
        help='Режим работы: train (обучение) или predict (прогноз)'
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
        run_demand_pipeline(data_path=args.data, output_dir=args.output)
    elif args.mode == 'predict':
        predict_demand(data_path=args.data)


if __name__ == "__main__":
    main()