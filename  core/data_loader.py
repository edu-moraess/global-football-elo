"""
Carregamento centralizado de dados com cache agressivo.
Toda computação pesada fica aqui. As páginas só consomem.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

from .elo_engine import compute_elo_history
from .poisson_engine import get_poisson_strengths

LOOKBACK_YEARS = 10

NAME_MAP = {
    'Czechoslovakia': 'Czech Republic',
    'Yugoslavia': 'Serbia',
    'Soviet Union': 'Russia',
    'West Germany': 'Germany',
    'East Germany': 'Germany'
}


@st.cache_data(show_spinner="Carregando base histórica...", ttl=3600)
def load_data():
    """Carrega e limpa o CSV de resultados."""
    df = pd.read_csv("data/results.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score'])
    df['home_team'] = df['home_team'].replace(NAME_MAP)
    df['away_team'] = df['away_team'].replace(NAME_MAP)
    return df


@st.cache_data(show_spinner="Computando ratings Elo...", ttl=1800)
def get_processed_data(df, lookback):
    """
    Computa Elo, histórico e forças Poisson.
    Retorna: elo_ratings, elo_history, attack, defense, league_avg
    """
    elo_ratings, elo_history = compute_elo_history(df, lookback)
    attack, defense, league_avg = get_poisson_strengths(df, lookback)
    return elo_ratings, elo_history, attack, defense, league_avg


@st.cache_data(show_spinner="Pré-computando métricas globais...", ttl=1800)
def get_global_metrics(elo_ratings, elo_history, teams):
    """
    Pré-computa todas as métricas avançadas para TODOS os times de uma vez.
    Evita recalcular métricas individualmente em loops nas páginas.
    """
    from .metrics_engine import (
        elo_volatility, elo_momentum, elo_efficiency,
        schedule_strength, team_consistency, get_confederation,
        form_index, tactical_profile
    )
    rows = []
    for team in teams:
        vol = elo_volatility(elo_history, team, window=10)
        mom = elo_momentum(elo_history, team)
        eff_real, eff_edge = elo_efficiency(elo_history, team, n_last=20)
        sched = schedule_strength(elo_history, team, n_last=10)
        consist = team_consistency(elo_history, team, n_last=20)
        conf = get_confederation(team)
        form = form_index(elo_history, team, n_last=5)
        tactical = tactical_profile(elo_history, team, n_last=20)
        rows.append({
            'Seleção': team,
            'Elo': elo_ratings[team],
            'Confederação': conf,
            'Volatilidade': vol,
            'Momentum': mom,
            'Eficiência Real': eff_real,
            'Edge Eficiência': -eff_edge if eff_edge is not np.nan else np.nan,
            'Força Agenda': sched,
            'Consistência': consist,
            'Forma (5j)': form,
            'xG Médio': tactical.get('xG', np.nan),
            'xGA Médio': tactical.get('xGA', np.nan),
            'Pressão Ofensiva': tactical.get('pressao', np.nan),
            'Resiliência': tactical.get('resiliencia', np.nan)
        })
    return pd.DataFrame(rows)
