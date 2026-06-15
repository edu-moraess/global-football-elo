# app.py - Global Football Intelligence (completo)
# Elo + Poisson alinhados com janela deslizante (últimos 10 anos)
# Inclui: evolução Elo, distribuição de gols, matriz de placares, head-to-head, forças, etc.

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# ================= CONFIGURAÇÕES GLOBAIS =================
LOOKBACK_YEARS = 10          # Mesma janela para Elo e Poisson
HOME_FACTOR_ATTACK = 1.2     # Vantagem de casa no ataque (Poisson)
HOME_FACTOR_DEFENSE = 0.9    # Vantagem de casa na defesa (Poisson)

# Configurações do Elo
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
    """
    Retorna (ratings_finais, DataFrame_historico)
    ratings_finais: dict {team: rating}
    historico: colunas date, home_team, away_team, home_rating_after, away_rating_after
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if lookback_years is not None:
        cutoff = df['date'].max() - timedelta(days=365 * lookback_years)
        df = df[df['date'] >= cutoff]
    
    teams = set(df['home_team']).union(set(df['away_team']))
    ratings = {team: INITIAL_ELO for team in teams}
    history = []
    
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
        
        history.append({
            'date': row['date'],
            'home_team': home,
            'away_team': away,
            'home_rating_after': ratings[home],
            'away_rating_after': ratings[away],
        })
    
    return ratings, pd.DataFrame(history)

# ================= FUNÇÕES DO POISSON =================
def get_poisson_strengths(df, lookback_years):
    """Retorna attack, defense, league_avg (todos Series indexados por time)"""
    cutoff = df['date'].max() - timedelta(days=365 * lookback_years)
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
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("data/results.csv")
    except FileNotFoundError:
        st.error("Arquivo 'data/results.csv' não encontrado. Usando dados de exemplo limitados.")
        # Dados sintéticos para demonstração
        np.random.seed(42)
        dates = pd.date_range('2015-01-01', periods=1500, freq='D')
        teams = ['Brazil', 'Argentina', 'Germany', 'France', 'Italy', 'England', 'Spain', 'Netherlands', 'Portugal', 'Uruguay']
        data = []
        for i in range(1500):
            home = np.random.choice(teams)
            away = np.random.choice([t for t in teams if t != home])
            data.append({
                'date': dates[i],
                'home_team': home,
                'away_team': away,
                'home_score': np.random.poisson(1.3),
                'away_score': np.random.poisson(1.1),
                'tournament': np.random.choice(['Friendly', 'FIFA World Cup', 'UEFA Euro', 'Copa América'], p=[0.6,0.2,0.1,0.1]),
                'neutral': np.random.choice([True, False], p=[0.3,0.7])
            })
        df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    if 'neutral' not in df.columns:
        df['neutral'] = False
    return df

# ================= INTERFACE PRINCIPAL =================
st.set_page_config(page_title="Global Football Intelligence", layout="wide")
st.title("🌍 Global Football Intelligence")
st.markdown(f"**Base histórica:** últimos {LOOKBACK_YEARS} anos | **Elo + Poisson** alinhados | *{len(load_data()):,} partidas*")

df = load_data()
teams = sorted(set(df['home_team']).union(set(df['away_team'])))

# Seleção de times
col1, col2, col3 = st.columns(3)
with col1:
    home_team = st.selectbox("Seleção da casa", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
with col2:
    away_team = st.selectbox("Seleção visitante", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)
with col3:
    neutral = st.checkbox("Campo neutro", value=False)

# Botão de predição
if st.button("🔮 Predizer", type="primary", use_container_width=True):
    with st.spinner("Calculando Elo e Poisson..."):
        # ========== ELO ==========
        elo_ratings, elo_history = compute_elo_history(df, lookback_years=LOOKBACK_YEARS)
        elo_home = elo_ratings.get(home_team, INITIAL_ELO)
        elo_away = elo_ratings.get(away_team, INITIAL_ELO)
        prob_elo_home = expected_score(elo_home, elo_away, neutral=neutral)
        
        # ========== POISSON ==========
        attack, defense, league_avg = get_poisson_strengths(df, LOOKBACK_YEARS)
        l_home, l_away, p_home, p_draw, p_away, prob_matrix = predict_poisson(
            home_team, away_team, neutral, attack, defense, league_avg
        )
        
        # ========== HEAD-TO-HEAD (últimos 10 anos) ==========
        cutoff = df['date'].max() - timedelta(days=365*LOOKBACK_YEARS)
        h2h = df[(df['date'] >= cutoff) & 
                 (((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
                  ((df['home_team'] == away_team) & (df['away_team'] == home_team)))]
        h2h_home_wins = len(h2h[(h2h['home_team'] == home_team) & (h2h['home_score'] > h2h['away_score'])])
        h2h_away_wins = len(h2h[(h2h['away_team'] == home_team) & (h2h['away_score'] > h2h['home_score'])])
        h2h_draws = len(h2h[h2h['home_score'] == h2h['away_score']])
        
        # ========== EXIBIÇÃO ==========
        # Linha 1: Cards de resumo
        st.subheader("📈 Resumo da Predição")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric(f"λ {home_team}", f"{l_home:.2f}")
        with col_b:
            st.metric(f"λ {away_team}", f"{l_away:.2f}")
        with col_c:
            st.metric(f"Elo {home_team}", f"{elo_home:.0f}")
        with col_d:
            st.metric(f"Elo {away_team}", f"{elo_away:.0f}")
        
        # Linha 2: Probabilidades lado a lado
        col_e, col_f = st.columns(2)
        with col_e:
            st.markdown("#### 🧠 Poisson Bivariado")
            st.write(f"**{home_team} vence:** {p_home:.1%}")
            st.write(f"**Empate:** {p_draw:.1%}")
            st.write(f"**{away_team} vence:** {p_away:.1%}")
        with col_f:
            st.markdown("#### 🏆 Elo (referência)")
            st.write(f"**{home_team} vence:** {prob_elo_home:.1%}")
            st.write(f"**{away_team} vence:** {1-prob_elo_home:.1%}")
            if abs(prob_elo_home - p_home) > 0.1:
                st.warning("⚠️ Divergência >10% entre Elo e Poisson – provável efeito de forma recente.")
        
        # Linha 3: Forças ofensivas/defensivas
        st.subheader("⚔️ Forças Relativas (últimos 10 anos)")
        attack_home = attack.get(home_team, 1.0)
        attack_away = attack.get(away_team, 1.0)
        defense_home = defense.get(home_team, 1.0)
        defense_away = defense.get(away_team, 1.0)
        col_g, col_h = st.columns(2)
        with col_g:
            st.write(f"**Ataque** {home_team}: {attack_home:.2f} (média = 1.0)")
            st.write(f"**Defesa** {home_team}: {defense_home:.2f} (menos é melhor)")
        with col_h:
            st.write(f"**Ataque** {away_team}: {attack_away:.2f}")
            st.write(f"**Defesa** {away_team}: {defense_away:.2f}")
        
        # Linha 4: Head-to-head recente
        if not h2h.empty:
            st.subheader("📋 Confrontos Diretos (últimos 10 anos)")
            st.write(f"{home_team} venceu {h2h_home_wins}x, {away_team} venceu {h2h_away_wins}x, empates {h2h_draws}x (total {len(h2h)} jogos)")
        
        # Linha 5: Matriz de placares + top 10
        st.subheader("⚽ Matriz de Probabilidade de Placar")
        # Heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=prob_matrix[:7, :7],
            x=[str(i) for i in range(7)],
            y=[str(i) for i in range(7)],
            colorscale="Viridis",
            text=np.round(prob_matrix[:7, :7], 3),
            texttemplate='%{text:.1%}',
            textfont={"size": 9}
        ))
        fig_heatmap.update_layout(
            title=f"Placar ({home_team} × {away_team})",
            xaxis_title=f"Gols {away_team}",
            yaxis_title=f"Gols {home_team}",
            width=500, height=500
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Tabela top 10 placares
        scores = [(f"{i}×{j}", prob_matrix[i, j]) for i in range(8) for j in range(8) if prob_matrix[i, j] > 0.005]
        scores.sort(key=lambda x: -x[1])
        st.subheader("🏅 10 placares mais prováveis")
        st.table(pd.DataFrame(scores[:10], columns=["Placar", "Probabilidade"]).assign(Probabilidade=lambda x: x["Probabilidade"].apply(lambda p: f"{p:.1%}")))
        
        # Linha 6: Evolução do Elo (gráfico)
        st.subheader("📈 Evolução do Rating Elo (últimos 10 anos)")
        # Extrair histórico dos dois times
        hist_home = elo_history[(elo_history['home_team'] == home_team) | (elo_history['away_team'] == home_team)].copy()
        hist_away = elo_history[(elo_history['home_team'] == away_team) | (elo_history['away_team'] == away_team)].copy()
        
        def get_rating_series(hist, team):
            ratings = []
            for _, row in hist.iterrows():
                if row['home_team'] == team:
                    ratings.append((row['date'], row['home_rating_after']))
                else:
                    ratings.append((row['date'], row['away_rating_after']))
            return pd.DataFrame(ratings, columns=['date', 'rating']).drop_duplicates(subset='date', keep='last')
        
        if not hist_home.empty:
            df_home = get_rating_series(hist_home, home_team)
            df_away = get_rating_series(hist_away, away_team)
            fig_elo = go.Figure()
            fig_elo.add_trace(go.Scatter(x=df_home['date'], y=df_home['rating'], mode='lines', name=home_team))
            fig_elo.add_trace(go.Scatter(x=df_away['date'], y=df_away['rating'], mode='lines', name=away_team))
            fig_elo.update_layout(title="Evolução do Elo", xaxis_title="Data", yaxis_title="Rating", height=400)
            st.plotly_chart(fig_elo, use_container_width=True)
        else:
            st.info("Histórico insuficiente para mostrar evolução do Elo.")
        
        # Linha 7: Distribuição de gols esperados (Poisson)
        st.subheader("📊 Distribuição de Gols Esperados")
        goals = np.arange(0, 6)
        prob_home_goals = poisson.pmf(goals, l_home)
        prob_away_goals = poisson.pmf(goals, l_away)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(x=goals, y=prob_home_goals, name=home_team, marker_color='blue'))
        fig_dist.add_trace(go.Bar(x=goals, y=prob_away_goals, name=away_team, marker_color='red'))
        fig_dist.update_layout(barmode='group', title="Probabilidade de número de gols", xaxis_title="Gols", yaxis_title="Probabilidade", height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Rodapé técnico
        st.caption(f"Modelos calibrados com dados de {LOOKBACK_YEARS} anos. Fator casa no Poisson: ataque×{HOME_FACTOR_ATTACK}, defesa×{HOME_FACTOR_DEFENSE}.")
else:
    st.info("👈 Selecione dois times e clique em 'Predizer' para ver a análise completa.")