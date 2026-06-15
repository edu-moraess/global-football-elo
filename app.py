# app.py - Dashboard Global Football Intelligence
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import plotly.graph_objects as go

# IMPORTANTE: Certifique-se que o arquivo elo_engine.py está no mesmo diretório
from elo_engine import compute_elo_history, expected_score, TOURNAMENT_WEIGHTS

st.set_page_config(page_title="Global Football Intelligence", layout="wide")
st.title("🌍 Global Football Intelligence")

# Parâmetro global: últimos 10 anos para alinhar Elo e Poisson
LOOKBACK_YEARS = 10
HOME_FACTOR_ATTACK = 1.2
HOME_FACTOR_DEFENSE = 0.9

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv("data/results.csv")
    df['date'] = pd.to_datetime(df['date'])
    if 'neutral' not in df.columns:
        df['neutral'] = False
    return df

@st.cache_data(ttl=3600)
def get_elo_ratings(_df, lookback):
    ratings, _ = compute_elo_history(_df, lookback_years=lookback)
    return ratings

@st.cache_data(ttl=3600)
def get_poisson_strengths(_df, lookback):
    cutoff = _df['date'].max() - timedelta(days=365*lookback)
    df_filtered = _df[_df['date'] >= cutoff]
    
    home_goals = df_filtered.groupby('home_team')['home_score'].sum()
    home_matches = df_filtered.groupby('home_team')['home_score'].count()
    away_goals = df_filtered.groupby('away_team')['away_score'].sum()
    away_matches = df_filtered.groupby('away_team')['away_score'].count()
    
    goals_scored = home_goals.add(away_goals, fill_value=0)
    matches = home_matches.add(away_matches, fill_value=0)
    avg_scored = (goals_scored / matches).fillna(0)
    
    goals_conceded_home = df_filtered.groupby('home_team')['away_score'].sum()
    goals_conceded_away = df_filtered.groupby('away_team')['home_score'].sum()
    goals_conceded = goals_conceded_home.add(goals_conceded_away, fill_value=0)
    avg_conceded = (goals_conceded / matches).fillna(0)
    
    total_goals = df_filtered['home_score'].sum() + df_filtered['away_score'].sum()
    total_matches = len(df_filtered)
    league_avg = total_goals / total_matches if total_matches > 0 else 1.0
    
    attack = avg_scored / league_avg
    defense = avg_conceded / league_avg
    return attack, defense, league_avg

def predict_poisson(home, away, neutral, attack, defense, league_avg):
    att_home = attack.get(home, 1.0)
    att_away = attack.get(away, 1.0)
    def_home = defense.get(home, 1.0)
    def_away = defense.get(away, 1.0)
    
    if not neutral:
        lambda_home = league_avg * att_home * def_away * HOME_FACTOR_ATTACK
        lambda_away = league_avg * att_away * def_home * HOME_FACTOR_DEFENSE
    else:
        lambda_home = league_avg * att_home * def_away
        lambda_away = league_avg * att_away * def_home
    
    max_goals = 10
    prob_matrix = np.outer(poisson.pmf(np.arange(max_goals+1), lambda_home),
                           poisson.pmf(np.arange(max_goals+1), lambda_away))
    p_home_win = np.sum(np.tril(prob_matrix, -1))
    p_away_win = np.sum(np.triu(prob_matrix, 1))
    p_draw = np.sum(np.diag(prob_matrix))
    return lambda_home, lambda_away, p_home_win, p_draw, p_away_win, prob_matrix

# Carregar dados
df = load_data()
teams = sorted(set(df['home_team']).union(set(df['away_team'])))

col1, col2, col3 = st.columns(3)
with col1:
    home_team = st.selectbox("Seleção da casa", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
with col2:
    away_team = st.selectbox("Seleção visitante", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)
with col3:
    neutral = st.checkbox("Campo neutro", value=False)

if st.button("🔮 Predizer"):
    with st.spinner("Processando..."):
        elo_ratings = get_elo_ratings(df, LOOKBACK_YEARS)
        attack, defense, league_avg = get_poisson_strengths(df, LOOKBACK_YEARS)
        l_home, l_away, p_home, p_draw, p_away, prob_matrix = predict_poisson(
            home_team, away_team, neutral, attack, defense, league_avg
        )
        elo_home = elo_ratings.get(home_team, 1500)
        elo_away = elo_ratings.get(away_team, 1500)
        prob_elo_home = expected_score(elo_home, elo_away, neutral=neutral)
        prob_elo_away = 1 - prob_elo_home
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📊 Poisson Bivariado")
            st.metric(f"λ {home_team}", f"{l_home:.2f}")
            st.metric(f"λ {away_team}", f"{l_away:.2f}")
            st.write(f"Vitória {home_team}: **{p_home:.1%}**")
            st.write(f"Empate: **{p_draw:.1%}**")
            st.write(f"Vitória {away_team}: **{p_away:.1%}**")
        with col_b:
            st.subheader("🏆 Elo (referência)")
            st.metric(f"{home_team}", f"{elo_home:.0f}")
            st.metric(f"{away_team}", f"{elo_away:.0f}")
            st.write(f"Probabilidade Elo: {home_team} {prob_elo_home:.1%} — {away_team} {prob_elo_away:.1%}")
        
        st.subheader("⚽ Placares mais prováveis")
        scores = [(f"{i}×{j}", prob_matrix[i,j]) for i in range(6) for j in range(6) if prob_matrix[i,j] > 0.005]
        scores.sort(key=lambda x: -x[1])
        st.table(pd.DataFrame(scores[:10], columns=["Placar", "Prob"]).assign(Prob=lambda x: x["Prob"].apply(lambda p: f"{p:.1%}")))