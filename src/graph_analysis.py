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
        
    # Цвета для кластеров (до 10 групп)
    CLUSTER_COLORS = [
        '#4E9AF1',  # 0 — синий       (Enterprise Backend / Java)
        '#F4A261',  # 1 — оранжевый   (Frontend / Web)
        '#2EC4B6',  # 2 — бирюзовый
        '#E63946',  # 3 — красный     (Python / ML)
        '#A8DADC',  # 4 — голубой     (DevOps / Cloud)
        '#FFD166',  # 5 — жёлтый
        '#C77DFF',  # 6 — фиолетовый
        '#06D6A0',  # 7 — зелёный
        '#EF476F',  # 8 — розовый
        '#118AB2',  # 9 — тёмно-синий
    ]

    def export_for_pyvis(self, output_path: str = 'reports/skill_graph.html'):
        """Экспорт графа в интерактивный HTML с правильными цветами и настройками."""
        if self.graph is None:
            return

        try:
            from pyvis.network import Network
        except ImportError:
            print("⚠️ Установите pyvis: pip install pyvis")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        net = Network(
            height='720px', width='100%',
            bgcolor='#1a1a2e', font_color='white',
            heading='',
        )

        # Максимальный размер узла для нормализации
        max_size = max((d.get('size', 1) for _, d in self.graph.nodes(data=True)), default=1)

        for node_id, data in self.graph.nodes(data=True):
            group  = data.get('group', 0)
            color  = self.CLUSTER_COLORS[group % len(self.CLUSTER_COLORS)]
            raw_sz = data.get('size', 10)
            # Нормализуем: от 15 до 60 пикселей
            norm_size = 15 + int((raw_sz / max_size) * 45)

            net.add_node(
                node_id,
                label=node_id.upper() if len(node_id) <= 4 else node_id.title(),
                color=color,
                size=norm_size,
                title=(
                    f"<b>{node_id}</b><br>"
                    f"Вакансий: {raw_sz}<br>"
                    f"Кластер: {group}"
                ),
                font={'size': 14, 'color': 'white'},
                borderWidth=2,
                borderWidthSelected=4,
            )

        # Максимальный вес ребра для нормализации толщины
        max_weight = max((d.get('weight', 1) for _, _, d in self.graph.edges(data=True)), default=1)

        for src, dst, data in self.graph.edges(data=True):
            weight = data.get('weight', 1)
            width  = 1 + int((weight / max_weight) * 8)
            net.add_edge(
                src, dst,
                value=weight,
                width=width,
                title=f"{src} ↔ {dst}: {weight} совместных упоминаний",
                color={'opacity': 0.5},
            )

        # Физика: барнс-хат для больших графов (быстро и красиво)
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 180,
              "springConstant": 0.04,
              "damping": 0.12,
              "avoidOverlap": 0.3
            },
            "stabilization": {
              "enabled": true,
              "iterations": 300,
              "updateInterval": 25
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "hideEdgesOnDrag": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "edges": {
            "smooth": { "type": "continuous" }
          }
        }
        """)

        net.save_graph(output_path)
        print(f"   ✅ Интерактивный граф сохранен: {output_path}")
        
    def export_graph_data(self, output_path: str = 'models/graph_data.json'):
        """Экспорт данных графа для Streamlit в унифицированном формате."""
        if self.graph is None:
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append({
                'id':    node_id,
                'label': node_id,
                'size':  data.get('size', 1),
                'group': data.get('group', 0),
            })

        links = []
        for src, dst, data in self.graph.edges(data=True):
            links.append({
                'source': src,
                'target': dst,
                'weight': data.get('weight', 1),
            })

        export = {'nodes': nodes, 'links': links}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export, f, ensure_ascii=False, indent=2)
        print(f"   ✅ Данные графа сохранены: {output_path} ({len(nodes)} узлов, {len(links)} рёбер)")

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
