# app.py
# Dashboard Global Football Intelligence – com Elo e Poisson alinhados (últimos 10 anos)

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Importa as funções do elo_engine ajustado
from elo_engine import compute_elo_history, expected_score, TOURNAMENT_WEIGHTS

# ================= CONFIGURAÇÕES =================
LOOKBACK_YEARS = 10   # Mesma janela para Elo e Poisson
HOME_FACTOR_ATTACK = 1.2   # Ajuste de casa para o ataque no Poisson
HOME_FACTOR_DEFENSE = 0.9  # Ajuste de casa para a defesa no Poisson

# ================= CARREGAMENTO DE DADOS =================
@st.cache_data
def load_data():
    df = pd.read_csv("data/results.csv")
    df['date'] = pd.to_datetime(df['date'])
    if 'neutral' not in df.columns:
        df['neutral'] = False
    return df

@st.cache_data
def get_elo_ratings(df, lookback_years):
    ratings, _ = compute_elo_history(df, lookback_years=lookback_years)
    return ratings

@st.cache_data
def get_poisson_strengths(df, lookback_years):
    """Calcula forças ofensivas/defensivas para os últimos lookback_years"""
    df = df.copy()
    cutoff = df['date'].max() - timedelta(days=365*lookback_years)
    df = df[df['date'] >= cutoff]
    
    # Gols marcados e sofridos por time
    home_goals = df.groupby('home_team')['home_score'].sum()
    home_matches = df.groupby('home_team')['home_score'].count()
    away_goals = df.groupby('away_team')['away_score'].sum()
    away_matches = df.groupby('away_team')['away_score'].count()
    
    goals_scored = home_goals.add(away_goals, fill_value=0)
    matches = home_matches.add(away_matches, fill_value=0)
    avg_scored = (goals_scored / matches).fillna(0)
    
    goals_conceded_home = df.groupby('home_team')['away_score'].sum()
    goals_conceded_away = df.groupby('away_team')['home_score'].sum()
    goals_conceded = goals_conceded_home.add(goals_conceded_away, fill_value=0)
    avg_conceded = (goals_conceded / matches).fillna(0)
    
    total_goals = df['home_score'].sum() + df['away_score'].sum()
    total_matches = len(df)
    league_avg = total_goals / total_matches if total_matches > 0 else 1.0
    
    attack = avg_scored / league_avg
    defense = avg_conceded / league_avg
    return attack, defense, league_avg

def predict_poisson(home, away, neutral, attack, defense, league_avg):
    """Retorna lambda_home, lambda_away e probabilidades de resultado"""
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
    prob_home_win = np.sum(np.tril(prob_matrix, -1))
    prob_away_win = np.sum(np.triu(prob_matrix, 1))
    prob_draw = np.sum(np.diag(prob_matrix))
    
    return lambda_home, lambda_away, prob_home_win, prob_draw, prob_away_win, prob_matrix

# ================= INTERFACE STREAMLIT =================
st.set_page_config(page_title="Global Football Intelligence", layout="wide")
st.title("🌍 Global Football Intelligence")
st.markdown(f"*Base histórica: últimos {LOOKBACK_YEARS} anos | Modelo Poisson + Elo alinhados*")

df = load_data()

# Seleção de times
teams = sorted(set(df['home_team']).union(set(df['away_team'])))
col1, col2, col3 = st.columns(3)
with col1:
    home_team = st.selectbox("Seleção da casa", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
with col2:
    away_team = st.selectbox("Seleção visitante", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)
with col3:
    neutral = st.checkbox("Campo neutro", value=False)

# Botão para predizer
if st.button("🔮 Predizer", type="primary"):
    with st.spinner("Calculando Elo e Poisson..."):
        # Carrega ratings Elo (últimos LOOKBACK_YEARS)
        elo_ratings = get_elo_ratings(df, LOOKBACK_YEARS)
        # Carrega forças Poisson (últimos LOOKBACK_YEARS)
        attack, defense, league_avg = get_poisson_strengths(df, LOOKBACK_YEARS)
        
        # Predição Poisson
        l_home, l_away, p_home, p_draw, p_away, prob_matrix = predict_poisson(
            home_team, away_team, neutral, attack, defense, league_avg
        )
        
        # Probabilidade Elo (para referência)
        elo_home = elo_ratings.get(home_team, 1500)
        elo_away = elo_ratings.get(away_team, 1500)
        prob_elo_home = expected_score(elo_home, elo_away, neutral=neutral)
        prob_elo_away = 1 - prob_elo_home
        
        # Exibição dos resultados
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📊 Modelo de Predição (Poisson Bivariado)")
            st.metric(f"λ {home_team}", f"{l_home:.2f}")
            st.metric(f"λ {away_team}", f"{l_away:.2f}")
            st.write(f"**Vitória {home_team}:** {p_home:.1%}")
            st.write(f"**Empate:** {p_draw:.1%}")
            st.write(f"**Vitória {away_team}:** {p_away:.1%}")
        
        with col_b:
            st.subheader("🏆 Rating Elo (referência)")
            st.metric(f"{home_team} (Elo)", f"{elo_home:.0f}")
            st.metric(f"{away_team} (Elo)", f"{elo_away:.0f}")
            st.write(f"**Prob. implícita Elo:** {home_team} {prob_elo_home:.1%} — {away_team} {prob_elo_away:.1%}")
            if abs(prob_elo_home - p_home) > 0.1:
                st.warning("⚠️ Atenção: Elo e Poisson mostram favoritos diferentes. Isso pode indicar tendências recentes não capturadas pelo Elo (ajuste a janela ou o fator K).")
        
        # Matriz de placares mais prováveis
        st.subheader("⚽ Placares mais prováveis")
        scores = []
        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[1]):
                prob = prob_matrix[i, j]
                if prob > 0.005:
                    scores.append((f"{i}×{j}", prob))
        scores.sort(key=lambda x: -x[1])
        top10 = scores[:10]
        
        df_placares = pd.DataFrame(top10, columns=["Placar", "Probabilidade"])
        df_placares["Probabilidade"] = df_placares["Probabilidade"].apply(lambda x: f"{x:.1%}")
        st.table(df_placares)
        
        # Heatmap da matriz de probabilidades (opcional)
        st.subheader("🌡️ Matriz de Probabilidade de Placares")
        fig = go.Figure(data=go.Heatmap(
            z=prob_matrix[:6, :6],
            x=[f"{j}" for j in range(6)],
            y=[f"{i}" for i in range(6)],
            colorscale="Viridis",
            text=np.round(prob_matrix[:6, :6], 3),
            texttemplate='%{text:.1%}',
            textfont={"size": 10},
        ))
        fig.update_layout(
            title=f"Probabilidades de placar ({home_team} × {away_team})",
            xaxis_title=f"Gols {away_team}",
            yaxis_title=f"Gols {home_team}",
            width=600, height=500
        )
        st.plotly_chart(fig)
        
        # Rodapé explicativo
        st.caption(f"Nota: Ambos os modelos usam dados dos últimos {LOOKBACK_YEARS} anos. "
                   f"Fator de casa no Poisson: ataque×{HOME_FACTOR_ATTACK}, defesa×{HOME_FACTOR_DEFENSE}.")