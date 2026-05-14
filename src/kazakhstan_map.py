#!/usr/bin/env python3
"""
Модуль генерации интерактивной карты Казахстана
Визуализирует распределение IT-вакансий по городам.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

# Координаты городов Казахстана (lat, lon)
CITY_COORDS = {
    'Алматы':               (43.238949, 76.889709),
    'Астана':               (51.180000, 71.446000),
    'Шымкент':              (42.300000, 69.600000),
    'Караганда':            (49.800000, 73.100000),
    'Актобе':               (50.300000, 57.200000),
    'Тараз':                (42.900000, 71.400000),
    'Павлодар':             (52.300000, 76.950000),
    'Усть-Каменогорск':     (50.000000, 82.600000),
    'Семей':                (50.400000, 80.300000),
    'Атырау':               (47.100000, 51.900000),
    'Костанай':             (53.200000, 63.600000),
    'Кызылорда':            (44.900000, 65.500000),
    'Уральск':              (51.200000, 51.400000),
    'Петропавловск':        (54.900000, 69.200000),
    'Актау':                (43.700000, 51.200000),
    'Кокшетау':             (53.300000, 69.400000),
    'Талдыкорган':          (45.000000, 78.400000),
    'Туркестан':            (43.300000, 68.300000),
    'Экибастуз':            (51.700000, 75.300000),
    'Степногорск':          (52.400000, 71.900000),
    'Щучинск':              (52.940000, 70.190000),
    'Рудный':               (52.960000, 63.130000),
    'Лисаковск':            (52.530000, 62.500000),
    'Жанатас':              (43.900000, 68.200000),
    'Хромтау':              (50.260000, 58.440000),
    'Аксай (Казахстан)':    (51.180000, 53.000000),
    'Жанаозен':             (43.340000, 52.860000),
    'Отеген-Батыр':         (43.440000, 77.260000),
    'Курчатов (Казахстан)': (50.720000, 78.540000),
    'Житикара':             (52.190000, 61.720000),
}


def build_kazakhstan_map(
    df: pd.DataFrame,
    city_column: str = 'city',
    title: str = 'Распределение IT-вакансий по Казахстану',
) -> go.Figure:
    """
    Строит интерактивную bubble-карту Казахстана.

    Args:
        df: DataFrame с колонкой city
        city_column: название колонки с городом
        title: заголовок карты

    Returns:
        Plotly Figure
    """
    city_counts = df[city_column].value_counts().reset_index()
    city_counts.columns = ['city', 'vacancies']

    # Прикрепляем координаты
    city_counts['lat'] = city_counts['city'].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    city_counts['lon'] = city_counts['city'].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
    city_counts = city_counts.dropna(subset=['lat', 'lon'])

    total = city_counts['vacancies'].sum()
    city_counts['pct'] = (city_counts['vacancies'] / total * 100).round(1)

    # Нормализуем размер пузырей: 10..65
    vmax = city_counts['vacancies'].max()
    city_counts['bubble_size'] = 10 + (city_counts['vacancies'] / vmax * 55)

    fig = go.Figure()

    # Bubble-слой
    fig.add_trace(go.Scattergeo(
        lat=city_counts['lat'],
        lon=city_counts['lon'],
        mode='markers+text',
        marker=dict(
            size=city_counts['bubble_size'],
            color=city_counts['vacancies'],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(
                title='Вакансий',
                thickness=15,
                len=0.6,
            ),
            line=dict(width=1.5, color='white'),
            opacity=0.85,
            sizemode='diameter',
        ),
        text=city_counts['city'],
        textposition='top center',
        textfont=dict(size=11, color='white', family='Arial Black'),
        customdata=np.stack([
            city_counts['vacancies'],
            city_counts['pct'],
        ], axis=-1),
        hovertemplate=(
            '<b>%{text}</b><br>'
            'Вакансий: <b>%{customdata[0]:,}</b><br>'
            'Доля: <b>%{customdata[1]:.1f}%</b>'
            '<extra></extra>'
        ),
        name='',
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18, color='white'),
        ),
        geo=dict(
            scope='asia',
            resolution=50,
            center=dict(lat=48.5, lon=67.0),
            projection_scale=4.8,
            showland=True,
            landcolor='#1e3a5f',
            showocean=True,
            oceancolor='#0d1b2a',
            showlakes=True,
            lakecolor='#0d1b2a',
            showrivers=False,
            showcountries=True,
            countrycolor='#4a6fa5',
            countrywidth=1.5,
            showcoastlines=True,
            coastlinecolor='#4a6fa5',
            bgcolor='#0d1b2a',
        ),
        paper_bgcolor='#0d1b2a',
        plot_bgcolor='#0d1b2a',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=50, b=0),
        height=520,
    )

    return fig


def build_city_skills_chart(
    df: pd.DataFrame,
    skills_df: Optional[pd.DataFrame] = None,
    top_n_cities: int = 8,
) -> go.Figure:
    """
    Горизонтальный bar chart топ-городов + средняя зарплата.
    """
    city_stats = df.groupby('city').agg(
        vacancies=('city', 'count'),
        avg_salary=('salary_numeric', 'mean') if 'salary_numeric' in df.columns else ('city', 'count'),
    ).reset_index().sort_values('vacancies', ascending=False).head(top_n_cities)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=city_stats['city'],
        x=city_stats['vacancies'],
        orientation='h',
        name='Вакансий',
        marker=dict(
            color=city_stats['vacancies'],
            colorscale='Viridis',
            showscale=False,
        ),
        text=city_stats['vacancies'],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Вакансий: %{x:,}<extra></extra>',
    ))

    fig.update_layout(
        title='Топ городов по количеству IT-вакансий',
        xaxis_title='Количество вакансий',
        yaxis=dict(autorange='reversed'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=380,
        margin=dict(l=10, r=60, t=50, b=20),
    )

    return fig


def save_map_html(fig: go.Figure, output_path: str = 'reports/figures/kazakhstan_map.html') -> None:
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"   ✅ Карта сохранена: {output_path}")
