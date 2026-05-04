#!/usr/bin/env python3
"""
Модуль для графового анализа ИТ-навыков (Co-occurrence Network)
Позволяет находить связи между навыками и выделять экосистемы.
"""

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import itertools
from collections import Counter
import os
import json

class SkillGraphAnalyzer:
    """Анализатор графа ИТ-навыков на основе совместной встречаемости"""
    
    def __init__(self, min_cooccurrence: int = 3):
        """
        Args:
            min_cooccurrence: Минимальное количество совместных упоминаний для создания связи
        """
        self.min_cooccurrence = min_cooccurrence
        self.graph = None
        self.community_map = {}
        
    def build_graph(self, df: pd.DataFrame) -> nx.Graph:
        """Построение графа на основе столбца skills_extracted"""
        print("🕸️ Построение графа навыков...")
        
        if 'skills_extracted' not in df.columns:
            raise ValueError("В DataFrame нет колонки 'skills_extracted'")
            
        cooccurrences = Counter()
        skill_counts = Counter()
        
        for skills in df['skills_extracted'].dropna():
            if not isinstance(skills, list):
                try:
                    # Если навыки сохранены как строка
                    import ast
                    skills = ast.literal_eval(skills)
                except:
                    continue
                    
            if not isinstance(skills, list):
                continue
                
            # Оставляем только уникальные навыки в рамках одной вакансии
            skills = sorted(list(set(skills)))
            
            # Подсчет индивидуальных упоминаний
            for skill in skills:
                skill_counts[skill] += 1
                
            # Подсчет совместных упоминаний
            if len(skills) > 1:
                pairs = itertools.combinations(skills, 2)
                cooccurrences.update(pairs)
                
        # Создание графа
        G = nx.Graph()
        
        # Добавляем узлы с размером = частота
        for skill, count in skill_counts.items():
            if count >= self.min_cooccurrence:
                G.add_node(skill, size=count, label=skill)
                
        # Добавляем ребра
        for (s1, s2), weight in cooccurrences.items():
            if weight >= self.min_cooccurrence and s1 in G and s2 in G:
                G.add_edge(s1, s2, weight=weight, value=weight)
                
        self.graph = G
        print(f"   ✅ Граф построен: Узлов={G.number_of_nodes()}, Ребер={G.number_of_edges()}")
        return self.graph
        
    def detect_communities(self) -> dict:
        """Разбиение графа на кластеры (экосистемы навыков)"""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            print("⚠️ Граф пуст, кластеризация невозможна")
            return {}
            
        print("🔄 Обнаружение экосистем (кластеризация)...")
        communities = list(greedy_modularity_communities(self.graph, weight='weight'))
        
        community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i
                
        # --- Ручное объединение кластеров (маппинг) ---
        # Чтобы не зависеть от меняющихся ID, объединяем по ключевым навыкам
        def find_group(skill):
            return community_map.get(skill, None)
            
        merge_rules = [
            (['golang', 'go'], 'go'),         # Объединяем Go
            (['excel', 'ml'], 'ml'),          # Объединяем Data & ML
            (['jenkins', 'aws'], 'aws')       # Объединяем DevOps & Cloud
        ]
        
        for rule_skills, target_skill in merge_rules:
            target_group = find_group(target_skill)
            if target_group is not None:
                for skill in rule_skills:
                    source_group = find_group(skill)
                    if source_group is not None and source_group != target_group:
                        # Перемещаем все навыки из source_group в target_group
                        for node, grp in community_map.items():
                            if grp == source_group:
                                community_map[node] = target_group
                
        nx.set_node_attributes(self.graph, community_map, 'group')
        self.community_map = community_map
        
        # Пересчитываем количество уникальных кластеров после слияния
        unique_clusters = len(set(community_map.values()))
        print(f"   ✅ Найдено кластеров: {unique_clusters} (после ручного объединения)")
        return community_map
        
    def export_for_pyvis(self, output_path: str = 'reports/skill_graph.html'):
        """Экспорт графа в интерактивный HTML с помощью Pyvis"""
        if self.graph is None:
            return
            
        try:
            from pyvis.network import Network
        except ImportError:
            print("⚠️ Установите pyvis: pip install pyvis")
            return
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Настройка сети
        net = Network(height='700px', width='100%', bgcolor='#222222', font_color='white')
        net.from_nx(self.graph)
        
        # Визуальные настройки
        net.repulsion(node_distance=100, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)
        
        net.save_graph(output_path)
        print(f"   ✅ Интерактивный граф сохранен: {output_path}")
        
    def export_graph_data(self, output_path: str = 'models/graph_data.json'):
        """Экспорт данных графа для Streamlit"""
        if self.graph is None:
            return
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = nx.node_link_data(self.graph)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"   ✅ Данные графа сохранены: {output_path}")

if __name__ == "__main__":
    from preprocessing import VacancyPreprocessor
    from features import FeatureEngineer
    
    # Тестирование графового анализа
    preprocessor = VacancyPreprocessor()
    df = preprocessor.load_data('results.csv')
    df = preprocessor.preprocess(df)
    
    fe = FeatureEngineer()
    df['skills_extracted'] = df['clean_text'].apply(fe.extract_skills)
    
    ga = SkillGraphAnalyzer(min_cooccurrence=5)
    ga.build_graph(df)
    ga.detect_communities()
    ga.export_for_pyvis()
    ga.export_graph_data()
