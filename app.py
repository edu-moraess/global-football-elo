# app.py - World Cup Football Quant (Modular Version)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.engine import (
    compute_elo_history, get_poisson_strengths, 
    predict_match, monte_carlo, INITIAL_ELO
)

# ================= CONFIGURAÇÕES =================
LOOKBACK_YEARS = 10

@st.cache_data
def load_data():
    df = pd.read_csv("data/results.csv")
    df['date'] = pd.to_datetime(df['date'])
    # Remover jogos sem placar (jogos futuros ou cancelados)
    df = df.dropna(subset=['home_score', 'away_score'])
    return df

# ================= UI SETUP =================
st.set_page_config(page_title="Football Quant Intelligence", layout="wide", page_icon="⚽")

st.title("⚽ Football Quant Intelligence")
st.markdown("Análise estatística de seleções baseada em modelos **Elo** e **Poisson**.")

df = load_data()
try:
    elo_ratings, elo_history = compute_elo_history(df, LOOKBACK_YEARS)
    teams = sorted(list(elo_ratings.keys()))
except Exception as e:
    st.error(f"Erro ao processar dados de Elo: {e}")
    st.stop()

# Sidebar: Ranking
st.sidebar.header("🌍 Top 20 Global")
ranking_df = pd.DataFrame(list(elo_ratings.items()), columns=['Seleção', 'Elo']).sort_values('Elo', ascending=False).reset_index(drop=True)
ranking_df.index += 1
st.sidebar.dataframe(ranking_df.head(20), use_container_width=True)

# Main
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predição", "🎯 Análise de Placar", "📈 Histórico", "📊 Insights"])

with tab1:
    col_s1, col_s2, col_s3 = st.columns([2, 2, 1])
    home_team = col_s1.selectbox("Seleção A (Casa)", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    away_team = col_s2.selectbox("Seleção B (Fora)", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)
    neutral = col_s3.toggle("Campo Neutro", value=True)

    attack, defense, league_avg = get_poisson_strengths(df, LOOKBACK_YEARS)
    res = predict_match(home_team, away_team, neutral, attack, defense, league_avg, elo_ratings)
    mc_h, mc_d, mc_a = monte_carlo(res['l_home'], res['l_away'])
    
    # Metrics
    st.subheader("📊 Indicadores Chave")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Elo {home_team}", f"{res['elo_home']:.0f}")
    m2.metric(f"Elo {away_team}", f"{res['elo_away']:.0f}")
    m3.metric(f"xG {home_team}", f"{res['l_home']:.2f}")
    m4.metric(f"xG {away_team}", f"{res['l_away']:.2f}")

    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎲 Probabilidades")
        fig_pie = px.pie(
            values=[mc_h, mc_d, mc_a], 
            names=[home_team, "Empate", away_team],
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        xp_h = (mc_h * 3) + (mc_d * 1)
        xp_a = (mc_a * 3) + (mc_d * 1)
        st.info(f"**Pontos Esperados (xP):** {home_team}: {xp_h:.2f} | {away_team}: {xp_a:.2f}")

    with c2:
        st.subheader("🛡️ Radar de Força")
        categories = ['Ataque', 'Defesa', 'Elo', 'Recent Form', 'Power']
        
        def get_stats(team, elo, att, defen):
            s_att = np.clip(att.get(team, 1.0) * 50, 20, 100)
            s_def = np.clip((2.5 - defen.get(team, 1.0)) * 40, 20, 100)
            s_elo = np.clip((elo - 1000) / 10, 0, 100)
            return [s_att, s_def, s_elo, 80, (s_att + s_def + s_elo)/3]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=get_stats(home_team, res['elo_home'], attack, defense), theta=categories, fill='toself', name=home_team))
        fig_radar.add_trace(go.Scatterpolar(r=get_stats(away_team, res['elo_away'], attack, defense), theta=categories, fill='toself', name=away_team))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

with tab2:
    st.subheader("🎯 Probabilidade de Placares Exatos")
    fig_heat = px.imshow(
        res['matrix'][:6, :6],
        labels=dict(x=f"Gols {away_team}", y=f"Gols {home_team}", color="Prob"),
        x=[str(i) for i in range(6)], y=[str(i) for i in range(6)],
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        scores = [(f"{i}-{j}", res['matrix'][i,j]) for i in range(6) for j in range(6)]
        scores.sort(key=lambda x: x[1], reverse=True)
        st.markdown("**Top 5 Placares Prováveis**")
        st.table(pd.DataFrame(scores[:5], columns=["Placar", "Probabilidade"]).assign(Probabilidade=lambda x: x['Probabilidade'].map('{:.1%}'.format)))
    
    with col_t2:
        st.markdown("**Probabilidades de Totais (Over/Under)**")
        prob_over_25 = np.sum([res['matrix'][i,j] for i in range(11) for j in range(11) if i+j > 2.5])
        prob_btts = np.sum([res['matrix'][i,j] for i in range(1, 11) for j in range(1, 11)])
        st.metric("Mais de 2.5 Gols", f"{prob_over_25:.1%}")
        st.metric("Ambas Marcam (BTTS)", f"{prob_btts:.1%}")

with tab3:
    st.subheader("📈 Evolução de Rating")
    h_h = elo_history[(elo_history['home_team'] == home_team) | (elo_history['away_team'] == home_team)].tail(50)
    h_a = elo_history[(elo_history['home_team'] == away_team) | (elo_history['away_team'] == away_team)].tail(50)
    
    def get_series(df_t, team):
        if df_t.empty: return []
        return [r['home_rating_after'] if r['home_team'] == team else r['away_rating_after'] for _, r in df_t.iterrows()]

    fig_evol = go.Figure()
    if not h_h.empty:
        fig_evol.add_trace(go.Scatter(x=h_h['date'], y=get_series(h_h, home_team), name=home_team, line=dict(width=3)))
    if not h_a.empty:
        fig_evol.add_trace(go.Scatter(x=h_a['date'], y=get_series(h_a, away_team), name=away_team, line=dict(width=3)))
    fig_evol.update_layout(xaxis_title="Data", yaxis_title="Elo Rating", hovermode="x unified")
    st.plotly_chart(fig_evol, use_container_width=True)

with tab4:
    st.subheader("📊 Insights das Seleções")
    
    def get_team_insights(team, data):
        recent = data[(data['home_team'] == team) | (data['away_team'] == team)].tail(20)
        wins = 0
        goals_scored = 0
        for _, row in recent.iterrows():
            if row['home_team'] == team:
                goals_scored += row['home_score']
                if row['home_score'] > row['away_score']: wins += 1
            else:
                goals_scored += row['away_score']
                if row['away_score'] > row['home_score']: wins += 1
        return wins / len(recent) if len(recent) > 0 else 0, goals_scored / len(recent) if len(recent) > 0 else 0

    win_rate_h, avg_g_h = get_team_insights(home_team, df)
    win_rate_a, avg_g_a = get_team_insights(away_team, df)
    
    c_i1, c_i2 = st.columns(2)
    with c_i1:
        st.info(f"**{home_team}** (Últimos 20 jogos)")
        st.write(f"Taxa de Vitória: {win_rate_h:.1%}")
        st.write(f"Média de Gols: {avg_g_h:.2f}")
        
    with c_i2:
        st.info(f"**{away_team}** (Últimos 20 jogos)")
        st.write(f"Taxa de Vitória: {win_rate_a:.1%}")
        st.write(f"Média de Gols: {avg_g_a:.2f}")

    st.divider()
    st.subheader("🏆 Distribuição de Elo Global")
    fig_dist = px.histogram(ranking_df, x="Elo", nbins=30, title="Distribuição de Força das Seleções")
    st.plotly_chart(fig_dist, use_container_width=True)

st.divider()
st.caption("© 2026 Football Quant Intelligence | Dados históricos e modelos estatísticos.")
