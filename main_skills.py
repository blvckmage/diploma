#!/usr/bin/env python3
"""
Главный скрипт для прогнозирования востребованности ИТ-навыков
Usage: python main_skills.py [--mode train] [--data results.csv] [--test-weeks 4]
"""

import argparse
import sys
import os
import joblib
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_close(path: str):
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"   Saved: {path}")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def run_classification(X_train, X_test, y_train_cls, y_test_cls,
                        output_dir: str) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, f1_score)

    print("\n" + "=" * 50)
    print("🎯 КЛАССИФИКАЦИЯ: demand_level (5 классов)")
    print("=" * 50)

    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    rows = []
    best_acc, best_clf, best_name = 0, None, ''
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train_cls)
        preds = clf.predict(X_test)
        acc  = accuracy_score(y_test_cls, preds)
        f1   = f1_score(y_test_cls, preds, average='weighted', zero_division=0)
        rows.append({'model': name, 'accuracy': round(acc, 4), 'f1_weighted': round(f1, 4)})
        print(f"   {name:<25} Acc={acc:.4f}  F1={f1:.4f}")
        if acc > best_acc:
            best_acc, best_clf, best_name = acc, clf, name

    cls_df = pd.DataFrame(rows)
    os.makedirs('reports', exist_ok=True)
    cls_df.to_csv('reports/classification_report.csv', index=False)
    print(f"   ✅ Сохранено: reports/classification_report.csv")

    # Confusion matrix для лучшей модели
    preds_best = best_clf.predict(X_test)
    cm = confusion_matrix(y_test_cls, preds_best)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix — {best_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    _save_close(f'{output_dir}/confusion_matrix.png')

    return cls_df


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def run_learning_curves(model, X_train, y_train, output_dir: str):
    from sklearn.model_selection import learning_curve

    print("\n📈 Кривые обучения...")
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=3, scoring='r2',
        train_sizes=np.linspace(0.2, 1.0, 8),
        n_jobs=-1
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train R²', color='steelblue')
    ax.fill_between(train_sizes,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.2, color='steelblue')
    ax.plot(train_sizes, val_scores.mean(axis=1),  'o-', label='CV R²',    color='coral')
    ax.fill_between(train_sizes,
                    val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.2, color='coral')
    ax.set_title('Кривые обучения (Gradient Boosting)')
    ax.set_xlabel('Размер обучающей выборки')
    ax.set_ylabel('R²')
    ax.legend()
    ax.grid(alpha=0.3)
    _save_close(f'{output_dir}/learning_curves.png')


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def run_shap(model, X_train, feature_cols, output_dir: str):
    print("\n🔍 SHAP — объяснение модели...")
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train,
                          feature_names=feature_cols,
                          show=False, plot_size=None)
        _save_close(f'{output_dir}/shap_summary.png')
        print(f"   ✅ SHAP сохранён: {output_dir}/shap_summary.png")
    except Exception as e:
        print(f"   ⚠️  SHAP недоступен: {e}")


# ---------------------------------------------------------------------------
# ARIMA + Baselines
# ---------------------------------------------------------------------------

def run_arima_and_baselines(analyzer: SkillAnalyzer, best_model,
                              feature_cols, output_dir: str):
    print("\n" + "=" * 50)
    print("📊 ARIMA + NAIVE BASELINES + ADF-ТЕСТ")
    print("=" * 50)

    try:
        from src.arima_forecast import (run_arima_comparison,
                                         run_adf_for_all_skills)

        weekly_demand = analyzer.weekly_demand
        top_skills = (
            weekly_demand.groupby('skill')['vacancy_count']
            .sum().nlargest(10).index.tolist()
        )

        # ADF тест
        adf_df = run_adf_for_all_skills(weekly_demand)
        adf_df.to_csv('reports/adf_results.csv', index=False)
        print(f"   ✅ ADF сохранён: reports/adf_results.csv")

        # ARIMA + Persistence + MA-4 + GB
        gb_feature_cols = [c for c in feature_cols if c in weekly_demand.columns]
        arima_df = run_arima_comparison(
            weekly_demand,
            top_skills=top_skills,
            n_test=4,
            gb_model=best_model.model if gb_feature_cols else None,
            feature_cols=gb_feature_cols if gb_feature_cols else None,
            output_path='reports/arima_comparison.csv',
        )

        # Визуализация сравнения
        if not arima_df.empty:
            metrics_cols = ['mae_persistence', 'mae_ma4', 'mae_arima']
            available = [c for c in metrics_cols if c in arima_df.columns]
            if available:
                fig, ax = plt.subplots(figsize=(11, 5))
                x = np.arange(len(arima_df))
                width = 0.25
                colors = ['#F4A261', '#4E9AF1', '#2EC4B6', '#06D6A0']
                for i, col in enumerate(available):
                    label = col.replace('mae_', '').upper()
                    ax.bar(x + i * width, arima_df[col], width,
                           label=f'MAE {label}', color=colors[i])
                ax.set_xticks(x + width)
                ax.set_xticklabels(arima_df['skill'], rotation=30, ha='right')
                ax.set_ylabel('MAE')
                ax.set_title('Сравнение моделей прогнозирования (MAE)')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                _save_close(f'{output_dir}/baseline_comparison.png')

        # Сохраняем недельный спрос
        weekly_demand.to_csv('reports/weekly_demand.csv', index=False)
        print(f"   ✅ weekly_demand.csv сохранён")

    except Exception as e:
        print(f"   ⚠️  ARIMA/Baselines ошибка: {e}")
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_skill_pipeline(
    data_path: str = 'results.csv',
    output_dir: str = 'reports/figures',
    n_test_weeks: int = 4,
):
    """Запуск пайплайна прогнозирования востребованности ИТ-навыков."""
    print("=" * 70)
    print("🚀 ПРОГНОЗИРОВАНИЕ ВОСТРЕБОВАННОСТИ ИТ-НАВЫКОВ")
    print("=" * 70)
    print(f"📅 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # ===== ЭТАП 1: ЗАГРУЗКА =====
    print("\n" + "=" * 50)
    print("📊 ЭТАП 1: ЗАГРУЗКА И АНАЛИЗ НАВЫКОВ")
    print("=" * 50)

    analyzer = SkillAnalyzer()
    analyzer.load_data(data_path)

    # ===== ЭТАП 1.5: ГРАФ =====
    print("\n" + "=" * 50)
    print("🕸️  ЭТАП 1.5: ГРАФ ЭКОСИСТЕМ НАВЫКОВ")
    print("=" * 50)
    try:
        from src.graph_analysis import SkillGraphAnalyzer
        from src.features import FeatureEngineer
        ga = SkillGraphAnalyzer(min_cooccurrence=5)
        if 'skills_extracted' not in analyzer.df.columns:
            fe = FeatureEngineer()
            analyzer.df['skills_extracted'] = analyzer.df['clean_text'].apply(fe.extract_skills)
        ga.build_graph(analyzer.df)
        ga.detect_communities()
        ga.export_for_pyvis(f'{output_dir}/skill_graph.html')
        ga.export_graph_data('models/graph_data.json')
    except Exception as e:
        print(f"⚠️  Ошибка графа: {e}")

    # ===== ЭТАП 2: ДАННЫЕ ДЛЯ МОДЕЛИРОВАНИЯ =====
    X, y_reg, feature_cols = analyzer.prepare_for_modeling(task='regression')
    _, y_cls, _            = analyzer.prepare_for_modeling(task='classification')

    print("\n📈 Топ-10 навыков:")
    trends = analyzer.analyze_demand_trends()
    print(trends.head(10).to_string())

    # ===== ЭТАП 2.5: K-MEANS =====
    print("\n" + "=" * 50)
    print("🔵 ЭТАП 2.5: K-MEANS КЛАСТЕРИЗАЦИЯ НАВЫКОВ")
    print("=" * 50)
    try:
        analyzer.cluster_skills_kmeans(output_path='reports/kmeans_skills.csv')
    except Exception as e:
        print(f"⚠️  K-Means ошибка: {e}")

    # ===== ЭТАП 3: ВРЕМЕННОЙ СПЛИТ =====
    print("\n" + "=" * 50)
    print("📊 ЭТАП 3: ХРОНОЛОГИЧЕСКИЙ СПЛИТ")
    print("=" * 50)

    X_train, X_test, y_train, y_test = analyzer.temporal_train_test_split(
        X, y_reg, n_test_weeks=n_test_weeks
    )
    _, _, y_train_cls, y_test_cls = analyzer.temporal_train_test_split(
        X, y_cls, n_test_weeks=n_test_weeks
    )

    # ===== ЭТАП 4: РЕГРЕССИЯ =====
    print("\n" + "=" * 50)
    print("🤖 ЭТАП 4: РЕГРЕССИЯ (Ridge / GBM / LightGBM)")
    print("=" * 50)

    comparator = RegressorComparator()
    comparator.add_model('Ridge', DemandRegressor('ridge'))
    comparator.add_model('Gradient Boosting', DemandRegressor('gradient_boosting'))

    try:
        from lightgbm import LGBMRegressor
        comparator.add_model('LightGBM', DemandRegressor('lightgbm'))
        print("   ✅ LightGBM подключён")
    except ImportError:
        print("   ⚠️  LightGBM не установлен")

    results = comparator.compare(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        cv=3
    )

    best_name, best_model = comparator.get_best_model()
    print(f"\n   Лучшая модель: {best_name}")

    evaluator = RegressionEvaluator(
        best_model.model,
        X_train.values, X_test.values,
        y_train.values, y_test.values,
        feature_names=feature_cols
    )
    metrics = evaluator.get_metrics()

    print(f"\n   📈 Метрики (тест = последние {n_test_weeks} недели):")
    print(f"      Train R²:   {metrics['train_r2']:.4f}")
    print(f"      Test  R²:   {metrics['test_r2']:.4f}")
    print(f"      Test  MAE:  {metrics['test_mae']:.4f}")
    print(f"      Test  RMSE: {metrics['test_rmse']:.4f}")
    print(f"      Test  MAPE: {metrics['test_mape']:.2f}%")

    # ===== ЭТАП 5: КЛАССИФИКАЦИЯ =====
    cls_df = run_classification(
        X_train.values, X_test.values,
        y_train_cls.values, y_test_cls.values,
        output_dir
    )

    # ===== ЭТАП 6: ВИЗУАЛИЗАЦИИ (регрессия) =====
    print("\n" + "=" * 50)
    print("📊 ЭТАП 6: ВИЗУАЛИЗАЦИИ")
    print("=" * 50)

    evaluator.plot_actual_vs_predicted(save_path=f'{output_dir}/skill_actual_vs_predicted.png')
    plt.close('all')
    evaluator.plot_residuals(save_path=f'{output_dir}/skill_residuals.png')
    plt.close('all')
    evaluator.plot_feature_importance(save_path=f'{output_dir}/skill_feature_importance.png')
    plt.close('all')

    # Demand distribution + top skills
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(y_reg, kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title('Распределение Demand Index')
    axes[0].set_xlabel('Demand Index')
    axes[0].set_ylabel('Частота')
    top_skills_df = trends.head(10)
    axes[1].barh(top_skills_df['skill'], top_skills_df['total_vacancies'], color='coral')
    axes[1].set_title('Топ-10 навыков')
    axes[1].set_xlabel('Количество вакансий')
    axes[1].invert_yaxis()
    plt.tight_layout()
    _save_close(f'{output_dir}/skill_demand_distribution.png')

    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    res = results.sort_values('Test_R2', ascending=True)
    x = np.arange(len(res))
    w = 0.35
    ax.barh(x - w/2, res['Train_R2'], w, label='Train R²', color='lightcoral')
    ax.barh(x + w/2, res['Test_R2'],  w, label='Test  R²', color='steelblue')
    ax.set_yticks(x)
    ax.set_yticklabels(res['Model'])
    ax.set_xlabel('R²')
    ax.set_title(f'Сравнение регрессоров (Test = {n_test_weeks} нед.)')
    ax.legend()
    ax.set_xlim(-0.1, 1.05)
    plt.tight_layout()
    _save_close(f'{output_dir}/skill_model_comparison.png')

    # ===== ЭТАП 7: LEARNING CURVES =====
    run_learning_curves(best_model.model, X_train.values, y_train.values, output_dir)

    # ===== ЭТАП 8: SHAP =====
    run_shap(best_model.model, X_train.values, feature_cols, output_dir)

    # ===== ЭТАП 9: ARIMA + BASELINES =====
    run_arima_and_baselines(analyzer, best_model, feature_cols, output_dir)

    # ===== ЭТАП 10: СОХРАНЕНИЕ =====
    print("\n" + "=" * 50)
    print("💾 ЭТАП 10: СОХРАНЕНИЕ АРТЕФАКТОВ")
    print("=" * 50)

    joblib.dump(best_model.model, 'models/skill_regressor.joblib')
    joblib.dump(feature_cols,     'models/skill_features.joblib')
    trends.to_csv('reports/skill_analysis.csv', index=False)

    metrics_df = pd.DataFrame([{
        'model': best_name,
        'test_weeks': n_test_weeks,
        **metrics
    }])
    metrics_df.to_csv('reports/model_metrics.csv', index=False)
    print(f"   ✅ reports/model_metrics.csv")
    print(f"   ✅ models/skill_regressor.joblib ({best_name})")

    print("\n✅ ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО!")
    return best_model, metrics, trends


def main():
    parser = argparse.ArgumentParser(
        description='Прогнозирование востребованности ИТ-навыков'
    )
    parser.add_argument('--mode',       type=str, default='train', choices=['train'])
    parser.add_argument('--data',       type=str, default='results.csv')
    parser.add_argument('--output',     type=str, default='reports/figures')
    parser.add_argument('--test-weeks', type=int, default=4,
                        help='Количество последних недель для тестовой выборки')

    args = parser.parse_args()

    if args.mode == 'train':
        run_skill_pipeline(
            data_path=args.data,
            output_dir=args.output,
            n_test_weeks=args.test_weeks,
        )


if __name__ == "__main__":
    main()
