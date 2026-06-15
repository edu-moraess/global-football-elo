# app.py - Global Football Intelligence (versão unificada, sem dependência externa)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ================= CONFIGURAÇÕES =================
LOOKBACK_YEARS = 10          # Mesma janela para Elo e Poisson
HOME_FACTOR_ATTACK = 1.2     # Ajuste de casa no ataque (Poisson)
HOME_FACTOR_DEFENSE = 0.9    # Ajuste de casa na defesa (Poisson)

# Configurações do Elo (incorporadas diretamente aqui)
INITIAL_ELO = 1500
K_FACTOR_BASE = 32
ELO_SCALE = 400
HOME_ADVANTAGE = 50

TOURNAMENT_WEIGHTS = {
    'FIFA World Cup': 60,
    'Copa América': 50,
    'UEFA Euro': 50,
    'FIFA World Cup qualification': 40,
    'UEFA Euro qualification': 35,
    'CONMEBOL qualification': 35,
    'AFC Asian Cup': 30,
    'Africa Cup of Nations': 30,
    'CONCACAF Gold Cup': 30,
    'OFC Nations Cup': 25,
    'Friendly': 20,
}

# ================= FUNÇÕES DO ELO =================
def goal_diff_multiplier(goal_diff):
    if goal_diff == 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return (11 + goal_diff) / 8.0

def expected_score(rating_a, rating_b, neutral=False):
    if neutral:
        diff = rating_b - rating_a
    else:
        diff = (rating_b - rating_a) + HOME_ADVANTAGE
    return 1.0 / (1.0 + 10 ** (diff / ELO_SCALE))

def compute_elo_history(df, lookback_years=10):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if lookback_years is not None:
        cutoff = df['date'].max() - timedelta(days=365*lookback_years)
        df = df[df['date'] >= cutoff]
    
    teams = set(df['home_team']).union(set(df['away_team']))
    ratings = {team: INITIAL_ELO for team in teams}
    
    for _, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        home_goals, away_goals = row['home_score'], row['away_score']
        tournament = row['tournament']
        neutral = row.get('neutral', False)
        
        r_home, r_away = ratings[home], ratings[away]
        exp_home = expected_score(r_home, r_away, neutral=neutral)
        exp_away = 1 - exp_home
        
        if home_goals > away_goals:
            res_home, res_away = 1, 0
        elif home_goals == away_goals:
            res_home, res_away = 0.5, 0.5
        else:
            res_home, res_away = 0, 1
        
        weight = TOURNAMENT_WEIGHTS.get(tournament, 20)
        gd = abs(home_goals - away_goals)
        k = K_FACTOR_BASE * (weight / 30.0) * goal_diff_multiplier(gd)
        
        ratings[home] = r_home + k * (res_home - exp_home)
        ratings[away] = r_away + k * (res_away - exp_away)
    
    return ratings

# ================= FUNÇÕES DO POISSON =================
def get_poisson_strengths(df, lookback_years):
    cutoff = df['date'].max() - timedelta(days=365*lookback_years)
    df_filtered = df[df['date'] >= cutoff]
    
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

# ================= CARREGAMENTO DE DADOS =================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/results.csv")
    except FileNotFoundError:
        # Fallback: cria dados de exemplo para demonstração
        st.error("Arquivo 'data/results.csv' não encontrado. Usando dados de exemplo limitados.")
        data = {
            'date': pd.date_range('2015-01-01', periods=1000),
            'home_team': np.random.choice(['Brazil', 'Argentina', 'Germany', 'France', 'Italy'], 1000),
            'away_team': np.random.choice(['Uruguay', 'England', 'Spain', 'Netherlands', 'Portugal'], 1000),
            'home_score': np.random.poisson(1.5, 1000),
            'away_score': np.random.poisson(1.2, 1000),
            'tournament': ['Friendly'] * 1000,
            'neutral': [False] * 1000
        }
        df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    if 'neutral' not in df.columns:
        df['neutral'] = False
    return df

# ================= INTERFACE STREAMLIT =================
st.set_page_config(page_title="Global Football Intelligence", layout="wide")
st.title("🌍 Global Football Intelligence")
st.markdown(f"*Base histórica: últimos {LOOKBACK_YEARS} anos | Elo e Poisson alinhados*")

df = load_data()
teams = sorted(set(df['home_team']).union(set(df['away_team'])))

col1, col2, col3 = st.columns(3)
with col1:
    home_team = st.selectbox("Seleção da casa", teams)
with col2:
    away_team = st.selectbox("Seleção visitante", teams)
with col3:
    neutral = st.checkbox("Campo neutro", value=False)

if st.button("🔮 Predizer", type="primary"):
    with st.spinner("Calculando..."):
        # Elo
        elo_ratings = compute_elo_history(df, lookback_years=LOOKBACK_YEARS)
        elo_home = elo_ratings.get(home_team, INITIAL_ELO)
        elo_away = elo_ratings.get(away_team, INITIAL_ELO)
        prob_elo_home = expected_score(elo_home, elo_away, neutral=neutral)
        
        # Poisson
        attack, defense, league_avg = get_poisson_strengths(df, LOOKBACK_YEARS)
        l_home, l_away, p_home, p_draw, p_away, prob_matrix = predict_poisson(
            home_team, away_team, neutral, attack, defense, league_avg
        )
        
        # Exibição dos resultados
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📊 Poisson Bivariado")
            st.metric(f"λ {home_team}", f"{l_home:.2f}")
            st.metric(f"λ {away_team}", f"{l_away:.2f}")
            st.write(f"**Vitória {home_team}:** {p_home:.1%}")
            st.write(f"**Empate:** {p_draw:.1%}")
            st.write(f"**Vitória {away_team}:** {p_away:.1%}")
        
        with col_b:
            st.subheader("🏆 Elo (referência)")
            st.metric(f"{home_team} (Elo)", f"{elo_home:.0f}")
            st.metric(f"{away_team} (Elo)", f"{elo_away:.0f}")
            st.write(f"**Probabilidade Elo:** {home_team} {prob_elo_home:.1%} — {away_team} {1-prob_elo_home:.1%}")
            if abs(prob_elo_home - p_home) > 0.1:
                st.warning("⚠️ Elo e Poisson mostram favoritos diferentes – provável devido a tendências recentes não capturadas pelo Elo.")
        
        st.subheader("⚽ Placares mais prováveis")
        scores = []
        for i in range(min(6, prob_matrix.shape[0])):
            for j in range(min(6, prob_matrix.shape[1])):
                prob = prob_matrix[i, j]
                if prob > 0.005:
                    scores.append((f"{i}×{j}", prob))
        scores.sort(key=lambda x: -x[1])
        top10 = scores[:10]
        st.table(pd.DataFrame(top10, columns=["Placar", "Probabilidade"]).assign(Probabilidade=lambda x: x["Probabilidade"].apply(lambda p: f"{p:.1%}")))
        
        # Heatmap
        st.subheader("🌡️ Matriz de Probabilidades (6×6)")
        fig = go.Figure(data=go.Heatmap(
            z=prob_matrix[:6, :6],
            x=[str(j) for j in range(6)],
            y=[str(i) for i in range(6)],
            colorscale="Viridis",
            text=np.round(prob_matrix[:6, :6], 3),
            texttemplate='%{text:.1%}',
            textfont={"size": 10}
        ))
        fig.update_layout(
            title=f"Placar ({home_team} × {away_team})",
            xaxis_title=f"Gols {away_team}",
            yaxis_title=f"Gols {home_team}",
            width=600, height=500
        )
        st.plotly_chart(fig)
        
        st.caption(f"Nota: Ambos os modelos usam dados dos últimos {LOOKBACK_YEARS} anos. "
                   f"Fator casa no Poisson: ataque×{HOME_FACTOR_ATTACK}, defesa×{HOME_FACTOR_DEFENSE}.")