#!/usr/bin/env python3
"""
Главный скрипт для сравнения NLP моделей
Сравнение TF-IDF, FastText и BERT для классификации вакансий
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.nlp_models import (
    TFIDFModel, FastTextModel, BERTModel, 
    NLPModelComparator, prepare_text_data,
    GENSIM_AVAILABLE, BERT_AVAILABLE
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def run_nlp_comparison(data_path: str = 'results_synthetic.csv'):
    """
    Сравнение NLP моделей для классификации вакансий
    """
    print("=" * 70)
    print("🤖 СРАВНЕНИЕ NLP МОДЕЛЕЙ ДЛЯ КЛАССИФИКАЦИИ ВАКАНСИЙ")
    print("=" * 70)
    print(f"📅 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ===== ЭТАП 1: ЗАГРУЗКА ДАННЫХ =====
    print("=" * 50)
    print("📊 ЭТАП 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("=" * 50)
    
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    print(f"✅ Загружено {len(df)} вакансий")
    
    # Подготовка текстовых данных
    df, y = prepare_text_data(df)
    
    print(f"   Текстовых полей: requirements + responsibilities (без title для избежания утечки)")
    print(f"   Классов: {len(np.unique(y))}")
    print(f"   Распределение классов:")
    class_dist = pd.Series(y).value_counts()
    for cls, count in class_dist.items():
        print(f"      {cls}: {count} ({count/len(y)*100:.1f}%)")
    
    # ===== ЭТАП 2: РАЗДЕЛЕНИЕ ДАННЫХ =====
    print("\n" + "=" * 50)
    print("📊 ЭТАП 2: РАЗДЕЛЕНИЕ ДАННЫХ")
    print("=" * 50)
    
    texts = df['full_text'].tolist()
    
    train_texts, test_texts, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(train_texts)} samples")
    print(f"   Test: {len(test_texts)} samples")
    
    # ===== ЭТАП 3: ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ =====
    print("\n" + "=" * 50)
    print("🔧 ЭТАП 3: ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ")
    print("=" * 50)
    
    comparator = NLPModelComparator()
    
    # 1. TF-IDF (всегда доступен)
    print("   ✅ TF-IDF + LogReg")
    comparator.add_model(TFIDFModel(max_features=5000))
    
    # 2. FastText (если gensim установлен)
    if GENSIM_AVAILABLE:
        print("   ✅ FastText")
        comparator.add_model(FastTextModel(vector_size=100))
    else:
        print("   ⚠️ FastText недоступен (установите gensim)")
    
    # 3. BERT (если transformers установлен)
    if BERT_AVAILABLE:
        print("   ✅ RuBERT (cointegrated/rubert-tiny2)")
        comparator.add_model(BERTModel(model_name='cointegrated/rubert-tiny2'))
    else:
        print("   ⚠️ BERT недоступен (установите transformers torch)")
    
    # ===== ЭТАП 4: СРАВНЕНИЕ МОДЕЛЕЙ =====
    print("\n" + "=" * 50)
    print("🚀 ЭТАП 4: СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 50)
    
    results = comparator.compare(train_texts, test_texts, y_train, y_test)
    
    # ===== ЭТАП 5: ЛУЧШАЯ МОДЕЛЬ =====
    print("\n" + "=" * 50)
    print("🏆 ЭТАП 5: ДЕТАЛЬНАЯ ОЦЕНКА ЛУЧШЕЙ МОДЕЛИ")
    print("=" * 50)
    
    best_name, best_model = comparator.get_best_model()
    print(f"   Лучшая модель: {best_name}")
    
    # Classification Report
    y_pred = best_model.predict(test_texts)
    class_names = sorted(np.unique(y_test))
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # ===== ЭТАП 6: ВИЗУАЛИЗАЦИИ =====
    print("\n" + "=" * 50)
    print("📊 ЭТАП 6: СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("=" * 50)
    
    os.makedirs('reports/figures', exist_ok=True)
    
    # Confusion Matrix для лучшей модели
    print("   Creating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {best_name}', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('reports/figures/nlp_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: reports/figures/nlp_confusion_matrix.png")
    
    # Сравнение моделей
    print("   Creating model comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    results_sorted = results.sort_values('Test_Accuracy', ascending=True)
    colors = ['steelblue' if m != best_name else 'gold' for m in results_sorted['Model']]
    axes[0].barh(results_sorted['Model'], results_sorted['Test_Accuracy'], color=colors)
    axes[0].set_xlabel('Test Accuracy')
    axes[0].set_title('Сравнение моделей по Accuracy')
    axes[0].set_xlim(0, 1)
    
    # F1 Score
    results_sorted = results.sort_values('Test_F1', ascending=True)
    colors = ['steelblue' if m != best_name else 'gold' for m in results_sorted['Model']]
    axes[1].barh(results_sorted['Model'], results_sorted['Test_F1'], color=colors)
    axes[1].set_xlabel('Test F1 Score')
    axes[1].set_title('Сравнение моделей по F1 Score')
    axes[1].set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig('reports/figures/nlp_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: reports/figures/nlp_model_comparison.png")
    
    # ===== ЭТАП 7: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ =====
    print("\n" + "=" * 50)
    print("💾 ЭТАП 7: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 50)
    
    # Сохранение результатов
    results.to_csv('reports/nlp_comparison_results.csv', index=False)
    print("   ✅ Results saved: reports/nlp_comparison_results.csv")
    
    # Сохранение лучшей модели
    import joblib
    joblib.dump(best_model, 'models/best_nlp_model.joblib')
    print("   ✅ Best model saved: models/best_nlp_model.joblib")
    
    # ===== ИТОГИ =====
    print("\n" + "=" * 70)
    print("✅ СРАВНЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 70)
    
    print(f"\n📊 Результаты:")
    print(f"   • Лучшая модель: {best_name}")
    print(f"   • Test Accuracy: {results[results['Model'] == best_name]['Test_Accuracy'].values[0]:.4f}")
    print(f"   • Test F1: {results[results['Model'] == best_name]['Test_F1'].values[0]:.4f}")
    
    print(f"\n📁 Сохранённые файлы:")
    print(f"   • reports/figures/nlp_confusion_matrix.png")
    print(f"   • reports/figures/nlp_model_comparison.png")
    print(f"   • reports/nlp_comparison_results.csv")
    print(f"   • models/best_nlp_model.joblib")
    
    print(f"\n📅 Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results, best_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Сравнение NLP моделей')
    parser.add_argument('--data', type=str, default='results_synthetic.csv',
                        help='Путь к файлу с данными')
    
    args = parser.parse_args()
    
    run_nlp_comparison(data_path=args.data)