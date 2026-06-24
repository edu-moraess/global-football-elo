"""DataLoader v2 — Cache centralizado."""
import streamlit as st
import pandas as pd
from .elo_engine import compute_elo_history, NAME_MAP
from .poisson_engine import fit_poisson_dc
from .metrics_engine import build_global_metrics

LOOKBACK_YEARS = 10


@st.cache_data(show_spinner="Carregando base histórica...", ttl=7200)
def load_results() -> pd.DataFrame:
    df = pd.read_csv("data/results.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score'])
    df['home_team'] = df['home_team'].replace(NAME_MAP)
    df['away_team'] = df['away_team'].replace(NAME_MAP)
    df['neutral']   = df['neutral'].astype(bool)
    return df


@st.cache_data(show_spinner="Carregando artilheiros...", ttl=7200)
def load_goalscorers() -> pd.DataFrame:
    df = pd.read_csv("data/goalscorers.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['home_team'] = df['home_team'].replace(NAME_MAP)
    df['away_team'] = df['away_team'].replace(NAME_MAP)
    return df


@st.cache_data(show_spinner="Computando ratings Elo...", ttl=3600)
def get_elo(df: pd.DataFrame, lookback: int = LOOKBACK_YEARS):
    return compute_elo_history(df, lookback)


@st.cache_data(show_spinner="Calibrando modelo Dixon-Coles...", ttl=3600)
def get_poisson(df: pd.DataFrame, lookback: int = LOOKBACK_YEARS):
    return fit_poisson_dc(df, lookback)


@st.cache_data(show_spinner="Pré-computando métricas globais...", ttl=3600)
def get_metrics(elo_ratings_items: tuple, history_key: int) -> pd.DataFrame:
    elo_ratings = dict(elo_ratings_items)
    history = st.session_state.get('_history_cache')
    if history is None:
        return pd.DataFrame()
    return build_global_metrics(elo_ratings, history)


def load_all(lookback: int = LOOKBACK_YEARS):
    df = load_results()
    elo_ratings, history = get_elo(df, lookback)
    st.session_state['_history_cache'] = history
    attack, defense, home_adv, rho, league_avg = get_poisson(df, lookback)
    metrics_df = get_metrics(tuple(sorted(elo_ratings.items())), id(history))
    return df, elo_ratings, history, attack, defense, home_adv, rho, league_avg, metrics_df