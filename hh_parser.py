#!/usr/bin/env python3
"""
Парсер вакансий с hh.kz (HeadHunter Kazakhstan)
Собирает информацию о вакансиях и сохраняет в CSV формат
"""

import requests
import csv
import time
import re
from datetime import datetime
from typing import Optional, List, Dict
import json

class HHParser:
    """Класс для парсинга вакансий с hh.kz"""
    
    BASE_URL = "https://api.hh.kz"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
        self.vacancies = []
        self.index = 0
    
    def search_vacancies(
        self,
        text: str,
        area: Optional[str] = None,
        professional_role: Optional[str] = None,
        experience: Optional[str] = None,
        employment: Optional[str] = None,
        schedule: Optional[str] = None,
        salary: Optional[int] = None,
        currency: str = "KZT",
        only_with_salary: bool = False,
        page: int = 0,
        per_page: int = 100
    ) -> Dict:
        """
        Поиск вакансий по параметрам
        
        Args:
            text: Поисковый запрос
            area: ID региона (например, '160' для Алматы, '159' для Астаны)
            professional_role: ID профессиональной роли
            experience: Опыт (noExperience, between1And3, between3And6, moreThan6)
            employment: Тип занятости (full, part, project, volunteer, probation)
            schedule: График работы (fullDay, shift, flexible, remote, flyInFlyOut)
            salary: Желаемая зарплата
            currency: Валюта (KZT, USD, EUR, RUB)
            only_with_salary: Только вакансии с указанной зарплатой
            page: Номер страницы
            per_page: Количество вакансий на странице (макс 100)
        
        Returns:
            Dict с результатами поиска
        """
        params = {
            'text': text,
            'page': page,
            'per_page': min(per_page, 100),
            'only_with_salary': str(only_with_salary).lower()
        }
        
        if area:
            params['area'] = area
        if professional_role:
            params['professional_role'] = professional_role
        if experience:
            params['experience'] = experience
        if employment:
            params['employment'] = employment
        if schedule:
            params['schedule'] = schedule
        if salary:
            params['salary'] = salary
            params['currency'] = currency
        
        try:
            response = self.session.get(f"{self.BASE_URL}/vacancies", params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Ошибка при поиске вакансий: {e}")
            return {'items': [], 'pages': 0, 'found': 0}
    
    def get_vacancy_details(self, vacancy_id: str) -> Optional[Dict]:
        """
        Получение детальной информации о вакансии
        
        Args:
            vacancy_id: ID вакансии
        
        Returns:
            Dict с детальной информацией или None
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/vacancies/{vacancy_id}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Ошибка при получении вакансии {vacancy_id}: {e}")
            return None
    
    def clean_html(self, text: Optional[str]) -> str:
        """Очистка HTML тегов из текста"""
        if not text:
            return ""
        # Удаляем HTML теги
        clean = re.sub(r'<[^>]+>', '', text)
        # Заменяем множественные пробелы и переносы
        clean = re.sub(r'\s+', ' ', clean)
        # Удаляем highlighttext теги специфичные для hh
        clean = re.sub(r'</?highlighttext>', '', clean)
        return clean.strip()
    
    def format_salary(self, salary_data: Optional[Dict]) -> str:
        """
        Форматирование зарплаты
        
        Args:
            salary_data: Данные о зарплате из API
        
        Returns:
            Строка с зарплатой
        """
        if not salary_data:
            return ""
        
        salary_from = salary_data.get('from')
        salary_to = salary_data.get('to')
        currency = salary_data.get('currency', '')
        
        if salary_from and salary_to:
            return f"{salary_from} - {salary_to} {currency}"
        elif salary_from:
            return f"от {salary_from} {currency}"
        elif salary_to:
            return f"до {salary_to} {currency}"
        
        return ""
    
    def format_date(self, date_str: Optional[str]) -> str:
        """
        Форматирование даты публикации
        
        Args:
            date_str: Дата в формате ISO
        
        Returns:
            Дата в формате DD-MM-YYYY
        """
        if not date_str:
            return ""
        
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime('%d-%m-%Y')
        except (ValueError, AttributeError):
            return date_str
    
    def extract_vacancy_data(self, vacancy: Dict) -> Dict:
        """
        Извлечение и форматирование данных вакансии
        
        Args:
            vacancy: Данные вакансии из API
        
        Returns:
            Dict с отформатированными данными
        """
        # Извлекаем название вакансии
        title = vacancy.get('name', '')
        
        # Зарплата
        salary = self.format_salary(vacancy.get('salary'))
        
        # Город
        area = vacancy.get('area', {})
        city = area.get('name', '') if area else ''
        
        # Компания
        employer = vacancy.get('employer', {})
        job = employer.get('name', '') if employer else ''
        
        # Дата публикации
        publish_date = self.format_date(vacancy.get('published_at'))
        
        # Требования и обязанности из snippet
        snippet = vacancy.get('snippet', {})
        requirements = self.clean_html(snippet.get('requirement', '')) if snippet else ''
        responsibilities = self.clean_html(snippet.get('responsibility', '')) if snippet else ''
        
        # График работы
        schedule_data = vacancy.get('schedule', {})
        schedule = schedule_data.get('name', '') if schedule_data else ''
        
        # Опыт
        experience_data = vacancy.get('experience', {})
        experience = experience_data.get('name', '') if experience_data else ''
        
        # Тип занятости
        employment_data = vacancy.get('employment', {})
        employment = employment_data.get('name', '') if employment_data else ''
        
        # URL вакансии
        url = vacancy.get('alternate_url', '') or vacancy.get('url', '')
        
        # Категория (определяем из professional_roles или из названия)
        professional_roles = vacancy.get('professional_roles', [])
        if professional_roles:
            job_category = professional_roles[0].get('name', '')
        else:
            job_category = self.determine_job_category(title)
        
        return {
            'index': self.index,
            'title': title,
            'salary': salary,
            'city': city,
            'job': job,
            'publish_date': publish_date,
            'requirements': requirements,
            'responsibilities': responsibilities,
            'schedule': schedule,
            'experience': experience,
            'employment': employment,
            'url': url,
            'Job': job_category
        }
    
    def determine_job_category(self, title: str) -> str:
        """
        Определение категории вакансии по названию
        
        Args:
            title: Название вакансии
        
        Returns:
            Категория вакансии
        """
        title_lower = title.lower()
        
        # Маппинг ключевых слов на категории
        categories = {
            'Data Analyst': ['data analyst', 'data аналитик', 'аналитик данных', 'big data аналитик'],
            'Data Scientist': ['data scientist', 'data science', 'дата сайентист'],
            'Designer': ['дизайнер', 'designer', 'ui/ux', 'ui', 'ux', 'веб-дизайнер'],
            'Developer': ['разработчик', 'developer', 'программист', 'frontend', 'backend', 'fullstack', 'java', 'python', 'php', 'react', 'vue'],
            'DevOps Engineer': ['devops', 'devops engineer', 'sre'],
            'HR Manager': ['hr', 'hr менеджер', 'менеджер по персоналу', 'рекрутер', 'специалист по персоналу'],
            'IT Specialist': ['it специалист', 'it specialist', 'системный администратор', 'технический специалист'],
            'ML Engineer': ['ml', 'machine learning', 'ml engineer', 'ml разработчик'],
            'Product Manager': ['product manager', 'продакт менеджер', 'менеджер продукта', 'product owner'],
            'Project Manager': ['project manager', 'проектный менеджер', 'руководитель проектов'],
            'Marketing': ['маркетолог', 'marketing', 'smm', 'digital marketing'],
            'Sales': ['продажи', 'sales', 'менеджер по продажам'],
            'Accountant': ['бухгалтер', 'accountant'],
            'Manager': ['менеджер', 'manager']
        }
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return category
        
        return 'Other'
    
    def parse_vacancies(
        self,
        search_text: str,
        area: Optional[str] = None,
        professional_role: Optional[str] = None,
        experience: Optional[str] = None,
        employment: Optional[str] = None,
        schedule: Optional[str] = None,
        max_pages: int = 10,
        per_page: int = 100,
        get_details: bool = False,
        delay: float = 0.5
    ) -> List[Dict]:
        """
        Основной метод парсинга вакансий
        
        Args:
            search_text: Поисковый запрос
            area: ID региона
            professional_role: ID профессиональной роли
            experience: Опыт работы
            employment: Тип занятости
            schedule: График работы
            max_pages: Максимальное количество страниц для парсинга
            per_page: Количество вакансий на странице
            get_details: Получать детальную информацию о каждой вакансии
            delay: Задержка между запросами (секунды)
        
        Returns:
            List[Dict] со списком вакансий
        """
        self.vacancies = []
        self.index = 0
        
        print(f"Начинаем парсинг вакансий по запросу: '{search_text}'")
        
        for page in range(max_pages):
            print(f"Парсинг страницы {page + 1}...")
            
            result = self.search_vacancies(
                text=search_text,
                area=area,
                professional_role=professional_role,
                experience=experience,
                employment=employment,
                schedule=schedule,
                page=page,
                per_page=per_page
            )
            
            items = result.get('items', [])
            if not items:
                print(f"Больше вакансий не найдено. Всего страниц: {page}")
                break
            
            for item in items:
                self.index += 1
                
                if get_details:
                    # Получаем детальную информацию
                    time.sleep(delay)
                    detailed = self.get_vacancy_details(item['id'])
                    if detailed:
                        vacancy_data = self.extract_vacancy_data(detailed)
                    else:
                        vacancy_data = self.extract_vacancy_data(item)
                else:
                    vacancy_data = self.extract_vacancy_data(item)
                
                self.vacancies.append(vacancy_data)
                print(f"  [{self.index}] {vacancy_data['title'][:50]}...")
            
            total_pages = result.get('pages', 0)
            if page >= total_pages - 1:
                print(f"Достигнута последняя страница ({total_pages})")
                break
            
            time.sleep(delay)
        
        print(f"\nПарсинг завершен. Собрано {len(self.vacancies)} вакансий.")
        return self.vacancies
    
    def save_to_csv(self, filename: str = 'results.csv', encoding: str = 'utf-8-sig') -> None:
        """
        Сохранение результатов в CSV файл
        
        Args:
            filename: Имя выходного файла
            encoding: Кодировка файла
        """
        if not self.vacancies:
            print("Нет данных для сохранения")
            return
        
        fieldnames = [
            'index', 'title', 'salary', 'city', 'job', 'publish_date',
            'requirements', 'responsibilities', 'schedule', 'experience',
            'employment', 'url', 'Job'
        ]
        
        try:
            with open(filename, 'w', newline='', encoding=encoding) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.vacancies)
            
            print(f"Данные сохранены в файл: {filename}")
        except IOError as e:
            print(f"Ошибка при сохранении файла: {e}")


def main():
    """Основная функция для демонстрации работы парсера"""
    parser = HHParser()
    
    # Пример 1: Поиск вакансий Data Analyst в Алматы
    print("=" * 60)
    print("Пример 1: Data Analyst в Алматы")
    print("=" * 60)
    
    vacancies = parser.parse_vacancies(
        search_text="Data Analyst",
        area="160",  # Алматы
        max_pages=2,
        per_page=50,
        get_details=False,  # Быстрый режим без деталей
        delay=0.3
    )
    
    parser.save_to_csv('data_analyst_almaty.csv')
    
    # Пример 2: Поиск вакансий Frontend Developer
    print("\n" + "=" * 60)
    print("Пример 2: Frontend Developer в Казахстане")
    print("=" * 60)
    
    parser2 = HHParser()
    vacancies2 = parser2.parse_vacancies(
        search_text="Frontend Developer",
        area="159",  # Астана
        max_pages=2,
        per_page=50,
        get_details=False,
        delay=0.3
    )
    
    parser2.save_to_csv('frontend_astana.csv')
    
    # Пример 3: Общий поиск по всем вакансиям с деталями
    print("\n" + "=" * 60)
    print("Пример 3: Все IT вакансии с детальной информацией")
    print("=" * 60)
    
    parser3 = HHParser()
    vacancies3 = parser3.parse_vacancies(
        search_text="программист OR разработчик OR developer",
        max_pages=1,
        per_page=20,
        get_details=True,  # Режим с детальной информацией
        delay=0.5
    )
    
    parser3.save_to_csv('it_vacancies_detailed.csv')


if __name__ == "__main__":
    main()