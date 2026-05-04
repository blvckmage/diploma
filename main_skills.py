#!/usr/bin/env python3
"""
Главный скрипт для прогнозирования востребованности ИТ-навыков
Usage: python main_skills.py [--mode train]
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.skill_analysis import SkillAnalyzer
from src.models import DemandRegressor, RegressorComparator
from src.evaluation import RegressionEvaluator

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def run_skill_pipeline(data_path: str = 'results.csv', output_dir: str = 'reports/figures'):
    """Запуск пайплайна прогнозирования востребованности ИТ-навыков (Регрессия)"""
    print("=" * 70)
    print("🚀 ПРОГНОЗИРОВАНИЕ ВОСТРЕБОВАННОСТИ ИТ-НАВЫКОВ (РЕГРЕССИЯ)")
    print("=" * 70)
    print(f"📅 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # ===== ЭТАП 1: АНАЛИЗ ВОСТРЕБОВАННОСТИ =====
    print("\n" + "=" * 50)
    print("📊 ЭТАП 1: АНАЛИЗ ВОСТРЕБОВАННОСТИ НАВЫКОВ")
    print("=" * 50)
    
    analyzer = SkillAnalyzer()
    analyzer.load_data(data_path)
    
    # ===== ЭТАП 1.5: АНАЛИЗ ГРАФОВ (UNSUPERVISED LEARNING) =====
    print("\n" + "=" * 50)
    print("🕸️ ЭТАП 1.5: АНАЛИЗ ЭКОСИСТЕМ НАВЫКОВ (ГРАФЫ)")
    print("=" * 50)
    try:
        from src.graph_analysis import SkillGraphAnalyzer
        ga = SkillGraphAnalyzer(min_cooccurrence=5)
        # Нам нужно убедиться, что навыки извлечены
        if 'skills_extracted' not in analyzer.df.columns:
            from src.features import FeatureEngineer
            fe = FeatureEngineer()
            analyzer.df['skills_extracted'] = analyzer.df['clean_text'].apply(fe.extract_skills)
            
        ga.build_graph(analyzer.df)
        ga.detect_communities()
        ga.export_for_pyvis(f'{output_dir}/skill_graph.html')
        ga.export_graph_data('models/graph_data.json')
    except Exception as e:
        print(f"⚠️ Ошибка при анализе графов: {e}")
    
    # Регрессия - предсказываем точное значение demand_index
    X, y, feature_cols = analyzer.prepare_for_modeling(task='regression')
    
    # Анализ трендов
    print("\n📈 Топ-10 самых востребованных навыков:")
    trends = analyzer.analyze_demand_trends()
    print(trends.head(10).to_string())
    
    # ===== ЭТАП 2: РАЗДЕЛЕНИЕ ДАННЫХ =====
    print("\n" + "=" * 50)
    print("📊 ЭТАП 2: РАЗДЕЛЕНИЕ ДАННЫХ")
    print("=" * 50)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    # ===== ЭТАП 3: СРАВНЕНИЕ МОДЕЛЕЙ =====
    print("\n" + "=" * 50)
    print("🤖 ЭТАП 3: СРАВНЕНИЕ МОДЕЛЕЙ (РЕГРЕССОРЫ)")
    print("=" * 50)
    
    comparator = RegressorComparator()
    comparator.add_model('Ridge', DemandRegressor('ridge'))
    comparator.add_model('Random Forest', DemandRegressor('random_forest'))
    comparator.add_model('Gradient Boosting', DemandRegressor('gradient_boosting'))
    
    results = comparator.compare(X_train.values, y_train.values, X_test.values, y_test.values, cv=5)
    
    # ===== ЭТАП 4: ЛУЧШАЯ МОДЕЛЬ =====
    print("\n" + "=" * 50)
    print("🏆 ЭТАП 4: ОЦЕНКА ЛУЧШЕЙ МОДЕЛИ")
    print("=" * 50)
    
    best_name, best_model = comparator.get_best_model()
    print(f"   Лучшая модель: {best_name}")
    
    evaluator = RegressionEvaluator(
        best_model.model, X_train.values, X_test.values, y_train.values, y_test.values,
        feature_names=feature_cols
    )
    
    metrics = evaluator.get_metrics()
    
    print(f"\n   📈 Метрики прогнозирования спроса (Регрессия):")
    print(f"      Train R2:  {metrics['train_r2']:.4f}")
    print(f"      Test R2:   {metrics['test_r2']:.4f}")
    print(f"      Test MAE:  {metrics['test_mae']:.4f}")
    print(f"      Test RMSE: {metrics['test_rmse']:.4f}")
    
    # ===== ЭТАП 5: ВИЗУАЛИЗАЦИИ =====
    print("\n" + "=" * 50)
    print("📊 ЭТАП 5: СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("=" * 50)
    
    evaluator.plot_actual_vs_predicted(save_path=f'{output_dir}/skill_actual_vs_predicted.png')
    plt.close()
    
    evaluator.plot_residuals(save_path=f'{output_dir}/skill_residuals.png')
    plt.close()
    
    evaluator.plot_feature_importance(save_path=f'{output_dir}/skill_feature_importance.png')
    plt.close()
    
    # Skill Demand Distribution Plot (Гистограмма для регрессии)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(y, kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title('Распределение индекса востребованности (Demand Index)', fontsize=12)
    axes[0].set_xlabel('Demand Index')
    axes[0].set_ylabel('Частота')
    
    top_skills = trends.head(10)
    axes[1].barh(top_skills['skill'], top_skills['total_vacancies'], color='coral')
    axes[1].set_title('Топ-10 востребованных навыков (по вакансиям)', fontsize=12)
    axes[1].set_xlabel('Количество вакансий (упоминаний)')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/skill_demand_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/skill_demand_distribution.png")
    
    # Model Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    results_sorted = results.sort_values('Test_R2', ascending=True)
    x = np.arange(len(results_sorted))
    width = 0.35
    ax.barh(x - width/2, results_sorted['Train_R2'], width, label='Train R2', color='lightcoral')
    ax.barh(x + width/2, results_sorted['Test_R2'], width, label='Test R2', color='steelblue')
    ax.set_yticks(x)
    ax.set_yticklabels(results_sorted['Model'])
    ax.set_xlabel('R2 Score')
    ax.set_title('Сравнение моделей: Спрос на навыки (R2)', fontsize=12)
    ax.legend()
    ax.set_xlim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/skill_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/skill_model_comparison.png")
    
    # ===== ЭТАП 6: СОХРАНЕНИЕ =====
    import joblib
    joblib.dump(best_model.model, 'models/skill_regressor.joblib')
    joblib.dump(feature_cols, 'models/skill_features.joblib')
    trends.to_csv('reports/skill_analysis.csv', index=False)
    
    print("\n✅ ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО!")
    return best_model, metrics, trends

def main():
    parser = argparse.ArgumentParser(description='Прогнозирование востребованности ИТ-навыков (Регрессия)')
    parser.add_argument('--mode', type=str, default='train', choices=['train'])
    parser.add_argument('--data', type=str, default='results.csv')
    parser.add_argument('--output', type=str, default='reports/figures')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_skill_pipeline(data_path=args.data, output_dir=args.output)

if __name__ == "__main__":
    main()
