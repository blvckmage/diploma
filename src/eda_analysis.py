#!/usr/bin/env python3
"""
EDA (Exploratory Data Analysis) — функции для анализа и визуализации данных.
Используются на странице EDA дашборда.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


def plot_weekly_trend(
    weekly_demand: pd.DataFrame,
    top_skills: Optional[List[str]] = None,
    n_skills: int = 8,
) -> go.Figure:
    """Линейный график спроса (demand_index) по неделям для топ навыков."""
    if top_skills is None:
        top_skills = (
            weekly_demand.groupby('skill')['vacancy_count']
            .sum().nlargest(n_skills).index.tolist()
        )

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, skill in enumerate(top_skills):
        sk = (weekly_demand[weekly_demand['skill'] == skill]
              .sort_values(['year', 'week']))
        fig.add_trace(go.Scatter(
            x=sk['year_week'],
            y=sk['demand_index'],
            mode='lines+markers',
            name=skill,
            line=dict(width=2, color=colors[i % len(colors)]),
            marker=dict(size=5),
            hovertemplate=f'<b>{skill}</b><br>Week: %{{x}}<br>Demand Index: %{{y:.3f}}<extra></extra>',
        ))

    fig.update_layout(
        title='Динамика Demand Index по неделям (топ навыки)',
        xaxis_title='Неделя',
        yaxis_title='Demand Index',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        height=420,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(tickangle=-45, gridcolor='rgba(255,255,255,0.08)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
    )
    return fig


def plot_salary_boxplot(df: pd.DataFrame) -> go.Figure:
    """Box-plot зарплат по категориям вакансий."""
    if 'salary_numeric' not in df.columns or 'Job' not in df.columns:
        return go.Figure()

    clean = df[df['salary_numeric'].notna() & (df['salary_numeric'] > 50_000)].copy()
    counts = clean['Job'].value_counts()
    top_jobs = counts[counts >= 10].index.tolist()
    clean = clean[clean['Job'].isin(top_jobs)]

    # Медиана для сортировки
    order = (clean.groupby('Job')['salary_numeric']
             .median().sort_values(ascending=False).index.tolist())

    fig = go.Figure()
    colors = px.colors.qualitative.Pastel
    for i, job in enumerate(order):
        vals = clean[clean['Job'] == job]['salary_numeric']
        fig.add_trace(go.Box(
            y=vals,
            name=job,
            boxmean='sd',
            marker_color=colors[i % len(colors)],
            line_width=1.5,
        ))

    fig.update_layout(
        title='Распределение зарплат по категориям вакансий (KZT)',
        yaxis_title='Зарплата (KZT)',
        showlegend=False,
        height=440,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
        xaxis=dict(tickangle=-30),
    )
    return fig


def plot_correlation_heatmap(weekly_demand: pd.DataFrame) -> go.Figure:
    """Тепловая карта корреляций числовых признаков."""
    num_cols = [
        'vacancy_count', 'avg_salary', 'unique_cities', 'unique_employers',
        'vacancy_score', 'salary_score', 'employer_score', 'geo_score',
        'growth_score', 'competition_score', 'demand_index',
    ]
    cols = [c for c in num_cols if c in weekly_demand.columns]
    corr = weekly_demand[cols].corr().round(2)

    # Красивые подписи
    labels = {
        'vacancy_count': 'Vacancy Count', 'avg_salary': 'Avg Salary',
        'unique_cities': 'Unique Cities', 'unique_employers': 'Unique Employers',
        'vacancy_score': 'Vacancy Score', 'salary_score': 'Salary Score',
        'employer_score': 'Employer Score', 'geo_score': 'Geo Score',
        'growth_score': 'Growth Score', 'competition_score': 'Competition Score',
        'demand_index': 'Demand Index',
    }
    tick_labels = [labels.get(c, c) for c in corr.columns]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=tick_labels,
        y=tick_labels,
        colorscale='RdBu',
        zmid=0,
        text=corr.values,
        texttemplate='%{text:.2f}',
        textfont=dict(size=10),
        hoverongaps=False,
        colorbar=dict(title='r', thickness=12),
    ))

    fig.update_layout(
        title='Матрица корреляций признаков Demand Index',
        height=480,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=11),
        xaxis=dict(tickangle=-40),
    )
    return fig


def plot_vacancy_count_distribution(weekly_demand: pd.DataFrame) -> go.Figure:
    """Гистограмма распределения vacancy_count."""
    fig = px.histogram(
        weekly_demand, x='vacancy_count', nbins=40,
        color_discrete_sequence=['#4E9AF1'],
        title='Распределение количества вакансий (за неделю, на навык)',
        labels={'vacancy_count': 'Количество вакансий', 'count': 'Частота'},
        marginal='box',
    )
    fig.update_layout(
        height=380,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
    )
    return fig


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """Bar chart пропущенных значений."""
    miss = (df.isnull().sum() / len(df) * 100).round(1)
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        return go.Figure()

    fig = px.bar(
        x=miss.values, y=miss.index, orientation='h',
        color=miss.values, color_continuous_scale='Reds',
        text=[f'{v:.1f}%' for v in miss.values],
        labels={'x': 'Доля пропусков (%)', 'y': 'Колонка'},
        title='Доля пропущенных значений по столбцам',
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=max(300, len(miss) * 35),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
    )
    return fig


def plot_adf_results(adf_df: pd.DataFrame) -> go.Figure:
    """Bar chart p-value ADF теста по навыкам."""
    df = adf_df.sort_values('p_value').head(20).copy()
    df['color'] = df['is_stationary'].map({True: '#06D6A0', False: '#EF476F'})

    fig = go.Figure(go.Bar(
        x=df['p_value'],
        y=df['skill'],
        orientation='h',
        marker_color=df['color'],
        text=[f"p={v:.3f}" for v in df['p_value']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>p-value: %{x:.4f}<extra></extra>',
    ))

    fig.add_vline(x=0.05, line_dash='dash', line_color='yellow',
                  annotation_text='α=0.05', annotation_position='top right')

    fig.update_layout(
        title='ADF-тест стационарности (зелёный = стационарный, p < 0.05)',
        xaxis_title='p-value',
        height=480,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)', range=[0, max(df['p_value'].max() * 1.1, 0.1)]),
        yaxis=dict(autorange='reversed'),
    )
    return fig
