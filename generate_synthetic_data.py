#!/usr/bin/env python3
"""
Генератор синтетических данных о вакансиях
Создаёт датасет с данными по месяцам для прогнозирования востребованности
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import re

# Настройки
np.random.seed(42)
random.seed(42)

# Параметры генерации
START_DATE = datetime(2024, 1, 1)  # Начало данных (январь 2024)
NUM_MONTHS = 24  # Количество месяцев (2 года данных для лучшего прогноза)
MIN_VACANCIES_PER_MONTH = 1000
MAX_VACANCIES_PER_MONTH = 2000

# Категории профессий (из оригинального датасета)
PROFESSIONS = [
    'Data Analyst',
    'Data Scientist', 
    'Developer',
    'Programmer',
    'QA Engineer',
    'DevOps Engineer',
    'IT Specialist',
    'HR Manager',
    'Recruiter',
    'Product Manager',
    'Designer',
    'ML Engineer',
    'Software Engineer',
    'Test Engineer'
]

# Шаблоны названий вакансий по профессиям
VACANCY_TITLES = {
    'Data Analyst': ['Data Analyst', 'Big Data аналитик', 'Data аналитик', 'Аналитик данных', 'BI аналитик', 'Junior Data Analyst', 'Senior Data Analyst'],
    'Data Scientist': ['Data Scientist', 'ML Engineer', 'Data Scientist Junior', 'Senior Data Scientist', 'ML аналитик'],
    'Developer': ['Fullstack Developer', 'Backend Developer', 'Frontend Developer', 'Senior Developer', 'Junior Developer', 'Software Developer'],
    'Programmer': ['Программист', 'Developer', 'Software Engineer', 'Junior Programmer', 'Senior Programmer', 'Python Developer', 'Java Developer'],
    'QA Engineer': ['QA Engineer', 'Тестировщик', 'QA Engineer Junior', 'Senior QA Engineer', 'Automation QA', 'Manual QA'],
    'DevOps Engineer': ['DevOps Engineer', 'DevOps', 'Senior DevOps', 'Cloud Engineer', 'SRE Engineer'],
    'IT Specialist': ['IT Specialist', 'IT инженер', 'Системный администратор', 'IT Support', 'Technical Support'],
    'HR Manager': ['HR Manager', 'HR менеджер', 'HR директор', 'HR Business Partner', 'HR Generalist'],
    'Recruiter': ['Recruiter', 'Рекрутер', 'IT Recruiter', 'Senior Recruiter', 'Technical Recruiter'],
    'Product Manager': ['Product Manager', 'Product Owner', 'Senior Product Manager', 'Junior Product Manager', 'Product Director'],
    'Designer': ['UI/UX Designer', 'Web Designer', 'Graphic Designer', 'Senior Designer', 'UX Designer', 'UI Designer'],
    'ML Engineer': ['ML Engineer', 'Machine Learning Engineer', 'AI Engineer', 'Senior ML Engineer', 'Junior ML Engineer'],
    'Software Engineer': ['Software Engineer', 'Senior Software Engineer', 'Junior Software Engineer', 'Software Developer'],
    'Test Engineer': ['Test Engineer', 'QA Engineer', 'Automation Engineer', 'Senior Test Engineer', 'Junior Test Engineer']
}

# Шаблоны зарплат по профессиям (в тенге)
SALARY_RANGES = {
    'Data Analyst': [(300000, 500000, 'Junior'), (500000, 800000, 'Middle'), (800000, 1200000, 'Senior')],
    'Data Scientist': [(400000, 600000, 'Junior'), (600000, 1000000, 'Middle'), (1000000, 1500000, 'Senior')],
    'Developer': [(350000, 550000, 'Junior'), (550000, 900000, 'Middle'), (900000, 1400000, 'Senior')],
    'Programmer': [(300000, 500000, 'Junior'), (500000, 850000, 'Middle'), (850000, 1300000, 'Senior')],
    'QA Engineer': [(280000, 450000, 'Junior'), (450000, 750000, 'Middle'), (750000, 1100000, 'Senior')],
    'DevOps Engineer': [(400000, 650000, 'Junior'), (650000, 1000000, 'Middle'), (1000000, 1600000, 'Senior')],
    'IT Specialist': [(200000, 350000, 'Junior'), (350000, 600000, 'Middle'), (600000, 900000, 'Senior')],
    'HR Manager': [(250000, 400000, 'Junior'), (400000, 700000, 'Middle'), (700000, 1000000, 'Senior')],
    'Recruiter': [(200000, 350000, 'Junior'), (350000, 600000, 'Middle'), (600000, 900000, 'Senior')],
    'Product Manager': [(350000, 600000, 'Junior'), (600000, 1000000, 'Middle'), (1000000, 1600000, 'Senior')],
    'Designer': [(280000, 450000, 'Junior'), (450000, 750000, 'Middle'), (750000, 1100000, 'Senior')],
    'ML Engineer': [(450000, 700000, 'Junior'), (700000, 1100000, 'Middle'), (1100000, 1700000, 'Senior')],
    'Software Engineer': [(350000, 550000, 'Junior'), (550000, 900000, 'Middle'), (900000, 1400000, 'Senior')],
    'Test Engineer': [(260000, 420000, 'Junior'), (420000, 700000, 'Middle'), (700000, 1050000, 'Senior')]
}

# Компании
COMPANIES = [
    'Казахтелеком', 'Beeline', 'Kaspi', 'Kaspi Bank', 'Halyk Bank',
    'Jusan Bank', 'ForteBank', 'Енергобанк', 'KPMG', 'Deloitte',
    'PwC', 'EY', 'McKinsey', 'Boston Consulting', 'Bain & Company',
    'Самрук-Казына', 'КазМунайГаз', 'KEGOC', 'KazMinerals', 'Kazatomprom',
    'Air Astana', 'Казахстанские железные дороги', 'KazPost', 'PM.KZ',
    '1С-Битрикс', 'Яндекс', 'Google Kazakhstan', 'Microsoft Kazakhstan',
    'Samsung', 'Apple', 'TikTok', 'Telegram', 'Meta',
    'Стартап А', 'Tech Startup B', 'FinTech Company', 'E-commerce Platform',
    'Банк ВТБ', 'Сбербанк', 'Альфа-Банк', 'Тинькофф', 'Райффайзен',
    'Центр развития города Алматы', 'Мобайл Телеком-Сервис', 'Kazakhstan Steel',
    'Казахстанско-Британский технический университет', 'НАО КазНТУ', 'KBTU'
]

# Города
CITIES = [
    'Алматы', 'Астана', 'Шымкент', 'Караганда', 'Актобе',
    'Тараз', 'Павлодар', 'Усть-Каменогорск', 'Семей', 'Костанай',
    'Уральск', 'Петропавловск', 'Кызылорда', 'Атырау', 'Актау'
]

# Обязанности (общие для всех профессий - без упоминания названия)
RESPONSIBILITIES_TEMPLATES = [
    'Выполнение поставленных задач в соответствии с требованиями проекта. Работа в команде.',
    'Участие в разработке и внедрении решений. Взаимодействие с командой и заказчиками.',
    'Анализ и обработка информации, подготовка отчётов и документации.',
    'Поддержка существующих систем и процессов, оптимизация работы.',
    'Разработка и внедрение новых решений, участие в проектах компании.',
    'Работа с внутренними и внешними заказчиками, сбор требований.',
    'Участие в планировании спринтов, оценка сроков выполнения задач.',
    'Проведение исследований и анализа данных для принятия решений.',
    'Документирование процессов и результатов работы.',
    'Взаимодействие с другими подразделениями компании.'
]

# Общие требования для ВСЕХ профессий (полностью пересекающиеся!)
# Модель должна учиться на сочетаниях признаков, а не на ключевых словах
REQUIREMENTS_TEMPLATES = {
    'Data Analyst': [
        'Опыт работы в IT-сфере от 1 года. Умение анализировать данные и находить закономерности.',
        'Высшее образование. Опыт работы с информацией и подготовки отчётов.',
        'Знание офисных программ, умение работать в команде.',
        'Опыт работы с документацией, внимательность к деталям.',
        'Коммуникабельность, ответственность, умение соблюдать дедлайны.'
    ],
    'Data Scientist': [
        'Опыт работы в IT-сфере от 1 года. Умение анализировать данные и находить закономерности.',
        'Высшее образование. Опыт работы с информацией и подготовки отчётов.',
        'Знание офисных программ, умение работать в команде.',
        'Опыт работы с документацией, внимательность к деталям.',
        'Коммуникабельность, ответственность, умение соблюдать дедлайны.'
    ],
    'Developer': [
        'Опыт работы в IT-сфере от 1 года. Умение анализировать данные и находить закономерности.',
        'Высшее образование. Опыт работы с информацией и подготовки отчётов.',
        'Знание офисных программ, умение работать в команде.',
        'Опыт работы с документацией, внимательность к деталям.',
        'Коммуникабельность, ответственность, умение соблюдать дедлайны.'
    ],
    'Programmer': [
        'Опыт работы в IT-сфере от 1 года. Умение анализировать данные и находить закономерности.',
        'Высшее образование. Опыт работы с информацией и подготовки отчётов.',
        'Знание офисных программ, умение работать в команде.',
        'Опыт работы с документацией, внимательность к деталям.',
        'Коммуникабельность, ответственность, умение соблюдать дедлайны.'
    ],
    'QA Engineer': [
        'Опыт работы в IT-сфере от 1 года. Умение анализировать данные и находить закономерности.',
        'Высшее образование. Опыт работы с информацией и подготовки отчётов.',
        'Знание офисных программ, умение работать в команде.',
        'Опыт работы с документацией, внимательность к деталям.',
        'Коммуникабельность, ответственность, умение соблюдать дедлайны.'
    ],
    'DevOps Engineer': [
        'Опыт работы в IT-сфере от 1 года. Умение анализировать данные и находить закономерности.',
        'Высшее образование. Опыт работы с информацией и подготовки отчётов.',
        'Знание офисных программ, умение работать в команде.',
        'Опыт работы с документацией, внимательность к деталям.',
        'Коммуникабельность, ответственность, умение соблюдать дедлайны.'
    ],
    'IT Specialist': [
        'Опыт работы в IT-сфере от 1 года. Умение анализировать данные и находить закономерности.',
        'Высшее образование. Опыт работы с информацией и подготовки отчётов.',
        'Знание офисных программ, умение работать в команде.',
        'Опыт работы с документацией, внимательность к деталям.',
        'Коммуникабельность, ответственность, умение соблюдать дедлайны.'
    ],
    'HR Manager': [
        'Опыт работы в кадровой сфере от 1 года. Умение работать с людьми.',
        'Высшее образование. Опыт работы с документами и информацией.',
        'Знание офисных программ, умение работать в команде.',
        'Опыт организации мероприятий, внимательность к деталям.',
        'Коммуникабельность, ответственность, умение соблюдать дедлайны.'
    ],
    'Recruiter': [
        'Опыт работы в кадровой сфере от 1 года. Умение работать с людьми.',
        'Высшее образование. Опыт работы с документами и информацией.',
        'Знание офисных программ, умение работать в команде.',
        'Опыт организации мероприятий, внимательность к деталям.',
        'Коммуникабельность, ответственность, умение соблюдать дедлайны.'
    ],
    'Product Manager': [
        'Опыт работы в IT-сфере от 1 года. Умение анализировать данные и находить закономерности.',
        'Высшее образование. Опыт работы с информацией и подготовки отчётов.',
        'Знание офисных программ, умение работать в команде.',
        'Опыт работы с документацией, внимательность к деталям.',
        'Коммуникабельность, ответственность, умение соблюдать дедлайны.'
    ],
    'Designer': [
        'Опыт работы в творческой сфере от 1 года. Умение работать в команде.',
        'Высшее образование. Опыт работы с информацией и документами.',
        'Знание офисных программ, креативность и внимательность.',
        'Опыт работы с документацией, внимательность к деталям.',
        'Коммуникабельность, ответственность, умение соблюдать дедлайны.'
    ],
    'ML Engineer': [
        'Опыт работы в IT-сфере от 1 года. Умение анализировать данные и находить закономерности.',
        'Высшее образование. Опыт работы с информацией и подготовки отчётов.',
        'Знание офисных программ, умение работать в команде.',
        'Опыт работы с документацией, внимательность к деталям.',
        'Коммуникабельность, ответственность, умение соблюдать дедлайны.'
    ],
    'Software Engineer': [
        'Опыт работы в IT-сфере от 1 года. Умение анализировать данные и находить закономерности.',
        'Высшее образование. Опыт работы с информацией и подготовки отчётов.',
        'Знание офисных программ, умение работать в команде.',
        'Опыт работы с документацией, внимательность к деталям.',
        'Коммуникабельность, ответственность, умение соблюдать дедлайны.'
    ],
    'Test Engineer': [
        'Опыт работы в IT-сфере от 1 года. Умение анализировать данные и находить закономерности.',
        'Высшее образование. Опыт работы с информацией и подготовки отчётов.',
        'Знание офисных программ, умение работать в команде.',
        'Опыт работы с документацией, внимательность к деталям.',
        'Коммуникабельность, ответственность, умение соблюдать дедлайны.'
    ]
}

# График работы
SCHEDULES = ['Полная занятость', 'Частичная занятость', 'Проектная работа', 'Стажировка', 'Удалённая работа']

# Опыт работы
EXPERIENCES = ['Нет опыта', 'От 1 года до 3 лет', 'От 3 до 6 лет', 'Более 6 лет']

# Занятость
EMPLOYMENTS = ['Полная занятость', 'Частичная занятость', 'Проектная работа', 'Стажировка']


def generate_salary(profession: str, experience: str) -> str:
    """Генерация зарплаты на основе профессии и опыта"""
    ranges = SALARY_RANGES.get(profession, SALARY_RANGES['IT Specialist'])
    
    if 'Нет опыта' in experience or 'Junior' in experience:
        min_sal, max_sal, _ = ranges[0]
    elif 'От 3 до 6' in experience or 'Senior' in experience:
        min_sal, max_sal, _ = ranges[2]
    else:
        min_sal, max_sal, _ = ranges[1]
    
    # Случайное колебание ±20%
    salary = random.randint(min_sal, max_sal)
    variation = int(salary * random.uniform(-0.2, 0.2))
    salary = max(min_sal, salary + variation)
    
    return f"от {salary - 50000} до {salary + 50000} ₸"


def generate_vacancy(vacancy_id: int, profession: str, date: datetime) -> dict:
    """Генерация одной вакансии"""
    title = random.choice(VACANCY_TITLES[profession])
    company = random.choice(COMPANIES)
    city = random.choice(CITIES)
    experience = random.choice(EXPERIENCES)
    
    # Зарплата (30% вакансий без зарплаты)
    if random.random() < 0.3:
        salary = ''
    else:
        salary = generate_salary(profession, experience)
    
    # Требования (уникальные для профессии)
    requirements = random.choice(REQUIREMENTS_TEMPLATES.get(profession, REQUIREMENTS_TEMPLATES['IT Specialist']))
    
    # Обязанности (общие для всех - БЕЗ названия профессии!)
    responsibilities = random.choice(RESPONSIBILITIES_TEMPLATES)
    
    schedule = random.choice(SCHEDULES)
    employment = random.choice(EMPLOYMENTS)
    
    # Форматирование даты
    publish_date = date.strftime('%d-%m-%Y')
    
    return {
        '': vacancy_id,
        'title': title,
        'salary': salary,
        'city': city,
        'job': company,
        'publish_date': publish_date,
        'requirements': requirements,
        'responsibilities': responsibilities,
        'schedule': schedule,
        'experience': experience,
        'employment': employment,
        'url': f'https://hh.ru/vacancy/{random.randint(80000000, 90000000)}',
        'Job': profession
    }


def generate_dataset():
    """Генерация полного датасета"""
    all_vacancies = []
    vacancy_id = 0
    
    # Сезонные колебания востребованности по профессиям
    # Формат: (профессия, базовый_коэффициент, сезонный_фактор)
    profession_trends = {
        'Data Analyst': (1.0, 0.3),      # Базовый спрос, средняя сезонность
        'Data Scientist': (0.8, 0.4),     # Ниже спрос, высокая сезонность
        'Developer': (1.5, 0.2),          # Высокий спрос, низкая сезонность
        'Programmer': (1.3, 0.25),        # Высокий спрос
        'QA Engineer': (0.9, 0.2),        # Средний спрос
        'DevOps Engineer': (0.7, 0.15),   # Ниже спрос, стабильный
        'IT Specialist': (0.6, 0.1),      # Низкий спрос, стабильный
        'HR Manager': (0.8, 0.35),        # Средний спрос, высокая сезонность
        'Recruiter': (0.9, 0.4),          # Средний спрос, высокая сезонность
        'Product Manager': (0.7, 0.2),    # Средний спрос
        'Designer': (0.85, 0.3),          # Средний спрос
        'ML Engineer': (0.5, 0.25),       # Ниже спрос, растущий тренд
        'Software Engineer': (0.75, 0.2), # Средний спрос
        'Test Engineer': (0.65, 0.15)     # Средний спрос
    }
    
    # Генерация по месяцам
    for month_offset in range(NUM_MONTHS):
        # Правильный расчёт месяца
        total_month = START_DATE.month + month_offset
        year = START_DATE.year + (total_month - 1) // 12
        month = ((total_month - 1) % 12) + 1
        
        # Базовое количество вакансий на месяц
        base_count = random.randint(MIN_VACANCIES_PER_MONTH, MAX_VACANCIES_PER_MONTH)
        
        # Сезонные колебания (меньше вакансий в январе, больше летом)
        seasonal_factor = {
            1: 0.7,   # Январь - меньше после праздников
            2: 0.85,
            3: 1.0,
            4: 1.1,
            5: 1.15,
            6: 1.1,
            7: 1.0,
            8: 1.05,
            9: 1.2,   # Сентябрь - сезон найма
            10: 1.15,
            11: 1.0,
            12: 0.8   # Декабрь - меньше перед праздниками
        }.get(month, 1.0)
        
        # Тренд роста рынка IT (5-10% в год)
        year_offset = month_offset // 12
        growth_factor = 1 + (0.05 * year_offset)
        
        # Финальное количество вакансий
        month_vacancy_count = int(base_count * seasonal_factor * growth_factor)
        
        # Распределение вакансий по профессиям
        profession_counts = {}
        total_weight = sum(trend[0] for trend in profession_trends.values())
        
        for prof, (base_weight, seasonality) in profession_trends.items():
            # Сезонные колебания для конкретной профессии
            prof_seasonal = 1 + seasonality * np.sin((month - 1) * np.pi / 6)
            weight = base_weight * prof_seasonal
            profession_counts[prof] = int(month_vacancy_count * weight / total_weight)
        
        # Корректировка, чтобы сумма равнялась month_vacancy_count
        diff = month_vacancy_count - sum(profession_counts.values())
        for _ in range(abs(diff)):
            prof = random.choice(PROFESSIONS)
            if diff > 0:
                profession_counts[prof] = profession_counts.get(prof, 0) + 1
            else:
                if profession_counts.get(prof, 0) > 0:
                    profession_counts[prof] -= 1
        
        # Генерация вакансий для месяца
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        num_days = days_in_month[month - 1]
        if year % 4 == 0 and month == 2:
            num_days = 29
        
        for prof, count in profession_counts.items():
            for _ in range(count):
                # Случайный день месяца
                day = random.randint(1, num_days)
                vacancy_date = datetime(year, month, day)
                
                vacancy = generate_vacancy(vacancy_id, prof, vacancy_date)
                all_vacancies.append(vacancy)
                vacancy_id += 1
    
    # Создание DataFrame
    df = pd.DataFrame(all_vacancies)
    
    print(f"✅ Сгенерировано {len(df)} вакансий")
    print(f"📅 Период: {df['publish_date'].min()} - {df['publish_date'].max()}")
    print(f"📊 Профессий: {df['Job'].nunique()}")
    
    # Статистика по месяцам
    df['date'] = pd.to_datetime(df['publish_date'], format='%d-%m-%Y')
    monthly_stats = df.groupby(df['date'].dt.to_period('M')).size()
    print(f"\n📊 Вакансий по месяцам:")
    for period, count in monthly_stats.items():
        print(f"   {period}: {count}")
    
    return df


def main():
    print("=" * 60)
    print("🔄 ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ")
    print("=" * 60)
    
    # Генерация датасета
    df = generate_dataset()
    
    # Сохранение
    output_path = 'results_synthetic.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Данные сохранены: {output_path}")
    
    # Статистика по профессиям
    print("\n📊 Распределение по профессиям:")
    prof_stats = df['Job'].value_counts()
    for prof, count in prof_stats.items():
        pct = count / len(df) * 100
        print(f"   {prof}: {count} ({pct:.1f}%)")
    
    return df


if __name__ == "__main__":
    main()