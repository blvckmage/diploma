# Прогнозный анализ востребованных профессий

## Описание проекта

**Тема диплома:** "Development of a predictive analytics model for predicting sought-after professions based on data from vacancies"

Система прогнозирования востребованности профессий на основе данных с HeadHunter.kz. Модель предсказывает уровень спроса на профессии (низкий/средний/высокий) для поддержки решений о выборе карьеры.

### Результаты

- **Задача:** Классификация уровня востребованности (5 классов)
- **Лучшая модель:** Gradient Boosting
- **Test Accuracy:** 55.6%
- **Test F1-Score:** 0.526
- **CV F1-Score:** 0.350
- **Профессий:** 14
- **Вакансий:** 1,754
- **Признаков:** 26 (без утечки данных)

---

## Архитектура проекта

### Компоненты системы

```
┌─────────────────────────────────────────────────────────────────┐
│                    ДАННЫЕ (results.csv)                         │
│                  1,754 вакансии с hh.kz                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ПРЕДОБРАБОТКА (preprocessing.py)                │
│  • Парсинг зарплат, дат, опыта                                  │
│  • Очистка HTML из описаний                                     │
│  • Создание текстовых признаков                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING (features.py)                  │
│  • TF-IDF векторы из описаний (50 признаков)                    │
│  • Бинарные признаки навыков (30 признаков)                     │
│  • Категориальное кодирование                                   │
│  • Масштабирование числовых признаков                          │
│  • Агрегированные статистики                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   МОДЕЛИ (models.py)                            │
│  • Logistic Regression                                          │
│  • Random Forest                                                │
│  • Gradient Boosting ← ЛУЧШАЯ                                   │
│  • KNN, XGBoost, LightGBM                                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ОЦЕНКА (evaluation.py)                        │
│  • Confusion Matrix                                             │
│  • Classification Report                                        │
│  • ROC Curves                                                   │
│  • SHAP анализ                                                  │
└─────────────────────────────────────────────────────────────────┘

---

## Структура проекта

```
DIPLOMA/
├── src/                        # Исходный код модулей
│   ├── __init__.py
│   ├── preprocessing.py        # Предобработка данных
│   ├── features.py             # Feature Engineering
│   ├── models.py               # ML модели
│   └── evaluation.py           # Оценка моделей
├── notebooks/                  # Jupyter notebooks
│   ├── feature_engineering.ipynb
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
├── reports/                    # Отчёты
│   ├── model_report.md
│   ├── final_report.md
│   └── figures/               # Визуализации
├── models/                     # Сохранённые модели
├── hh_parser.py               # Парсер вакансий
├── results.csv                # Датасет
├── requirements.txt           # Зависимости
└── README.md
```

---

## Установка

### 1. Клонировать репозиторий

```bash
git clone <repository-url>
cd DIPLOMA
```

### 2. Создать виртуальное окружение

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

---

## Использование

### Парсинг вакансий

```python
from hh_parser import HHParser

parser = HHParser()
vacancies = parser.parse_vacancies(
    search_text="Python Developer",
    area="160",  # Алматы
    max_pages=5,
    per_page=100
)
parser.save_to_csv('python_vacancies.csv')
```

### Предобработка данных

```python
from src import VacancyPreprocessor

preprocessor = VacancyPreprocessor()
df = preprocessor.load_data('results.csv')
df_processed = preprocessor.preprocess()
```

### Feature Engineering

```python
from src import FeatureEngineer

fe = FeatureEngineer()
X, y, features = fe.prepare_for_modeling(
    df_processed,
    target_column='Job',
    use_tfidf=True,
    use_skills=True
)
```

### Обучение модели

```python
from src import ProfessionClassifier, ModelComparator

# Одиночная модель
model = ProfessionClassifier('logistic')
model.fit(X_train, y_train)
metrics = model.evaluate(X_test, y_test)

# Сравнение моделей
comparator = ModelComparator()
comparator.add_model('Logistic', ProfessionClassifier('logistic'))
comparator.add_model('Random Forest', ProfessionClassifier('random_forest'))
results = comparator.compare(X_train, y_train, X_test, y_test)
```

### Оценка модели

```python
from src import ModelEvaluator

evaluator = ModelEvaluator(model.model, X_train, X_test, y_train, y_test,
                           feature_names=features, class_names=class_names)
evaluator.plot_confusion_matrix()
evaluator.plot_roc_curves()
```

---

## Доступные модели

| Модель | Ключ | Описание |
|--------|------|----------|
| Logistic Regression | `logistic` | Линейный классификатор |
| Random Forest | `random_forest` | Ансамбль деревьев |
| Gradient Boosting | `gradient_boosting` | Градиентный бустинг |
| K-Nearest Neighbors | `knn` | Метод k ближайших соседей |
| XGBoost | `xgboost` | Extreme Gradient Boosting |
| LightGBM | `lightgbm` | Light Gradient Boosting |

---

## Категории профессий

Модель классифицирует вакансии по 14 категориям:

1. Data Analyst
2. Data Scientist
3. DevOps Engineer
4. Java Developer
5. JavaScript Developer
6. ML Engineer
7. PHP Developer
8. Product Manager
9. Project Manager
10. Python Developer
11. QA Engineer
12. System Administrator
13. UI/UX Designer
14. Другое

---

## Целевая переменная

**Задача:** Прогнозирование уровня востребованности профессий

**Целевая переменная:** `demand_level` — комплексный индекс востребованности

### Методика расчёта индекса

**Формула:** `Demand_Index = Σ(w_i × norm(feature_i))`

| Компонента | Вес | Описание |
|------------|-----|----------|
| vacancy_score | 30% | Нормализованное количество вакансий |
| salary_score | 25% | Уровень зарплат относительно рынка |
| employer_score | 15% | Разнообразие работодателей |
| geo_score | 10% | Географический охват |
| growth_score | 10% | Тренд роста вакансий |
| competition_score | 10% | Обратный показатель конкуренции |

### Классификация (5 классов по квантилям)

| Класс | Описание | Квантиль |
|-------|----------|----------|
| 0 | Очень низкий | 0-20% |
| 1 | Низкий | 20-40% |
| 2 | Средний | 40-60% |
| 3 | Высокий | 60-80% |
| 4 | Очень высокий | 80-100% |

**Применение:** Помогает определить, какие профессии наиболее востребованы на рынке труда для принятия решений о выборе карьеры.

---

## Результаты прогнозирования востребованности

### Сравнение моделей

| Модель | Test Accuracy | Test F1 | CV F1 |
|--------|---------------|---------|-------|
| **Gradient Boosting** | **55.6%** | **0.526** | 0.350 |
| Logistic Regression | 33.3% | 0.289 | 0.343 |
| Random Forest | 22.2% | 0.178 | 0.340 |

### Топ-5 востребованных профессий

1. Programmer (700 вакансий)
2. Recruiter (264 вакансий)
3. HR Manager (253 вакансий)
4. Developer (200 вакансий)
5. IT Specialist (125 вакансий)

### Важные признаки

Топ-5 признаков для прогнозирования спроса:
1. profession_encoded (тип профессии)
2. prof_avg_vacancies (среднее количество вакансий)
3. vacancy_count_lag1 (предыдущая неделя)
4. week (неделя года)
5. avg_salary (средняя зарплата)

---

## Визуализации

Проект генерирует следующие визуализации:
- Confusion Matrix
- Classification Report Heatmap
- ROC Curves
- Learning Curves
- Class Distribution
- Feature Importance

---

## Технологии

- **Python 3.11+**
- **pandas, numpy** - обработка данных
- **scikit-learn** - машинное обучение
- **matplotlib, seaborn** - визуализация
- **BeautifulSoup, requests** - парсинг данных
- **XGBoost, LightGBM** - градиентный бустинг

---

## Лицензия

MIT License

---

## Автор

Дипломный проект - Разработка модели прогнозного анализа для предсказания востребованных профессий# diploma
