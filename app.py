# app.py - World Cup Football Quant (Modular Version)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards

from utils.engine import (
    compute_elo_history, get_poisson_strengths, 
    predict_match, monte_carlo, INITIAL_ELO
)
from utils.styles import apply_custom_styles, WC_COLORS

# ================= CONFIGURAÇÕES =================
LOOKBACK_YEARS = 10

@st.cache_data
def load_data():
    df = pd.read_csv("data/results.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

# ================= UI SETUP =================
st.set_page_config(page_title="WC Football Quant", layout="wide", page_icon="🏆")
apply_custom_styles()

st.markdown("<h1 class='main-header'>🏆 World Cup Football Quant Intelligence</h1>", unsafe_allow_html=True)

df = load_data()
elo_ratings, elo_history = compute_elo_history(df, LOOKBACK_YEARS)
teams = sorted(list(elo_ratings.keys()))

# Sidebar: Ranking
st.sidebar.markdown(f"<h2 style='color:{WC_COLORS['maroon']}'>🌍 Top 20 Global</h2>", unsafe_allow_html=True)
ranking_df = pd.DataFrame(list(elo_ratings.items()), columns=['Seleção', 'Elo']).sort_values('Elo', ascending=False).reset_index(drop=True)
ranking_df.index += 1
st.sidebar.dataframe(ranking_df.head(20), use_container_width=True)

# Main
tab1, tab2, tab3 = st.tabs(["🔮 Predição", "🎯 Análise de Placar", "📈 Histórico"])

with tab1:
    col_s1, col_s2, col_s3 = st.columns([2, 2, 1])
    home_team = col_s1.selectbox("Seleção A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    away_team = col_s2.selectbox("Seleção B", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)
    neutral = col_s3.toggle("Neutro", value=True)

    if st.button("🚀 Analisar Confronto", type="primary", use_container_width=True):
        attack, defense, league_avg = get_poisson_strengths(df, LOOKBACK_YEARS)
        res = predict_match(home_team, away_team, neutral, attack, defense, league_avg, elo_ratings)
        mc_h, mc_d, mc_a = monte_carlo(res['l_home'], res['l_away'])
        
        # Metrics
        st.markdown("### 📊 Indicadores Chave")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rating Elo", f"{res['elo_home']:.0f}", help=f"Força atual de {home_team}")
        m2.metric("Rating Elo", f"{res['elo_away']:.0f}", help=f"Força atual de {away_team}")
        m3.metric("Gols Esperados (xG)", f"{res['l_home']:.2f}")
        m4.metric("Gols Esperados (xG)", f"{res['l_away']:.2f}")
        style_metric_cards(background_color="#FFFFFF", border_left_color=WC_COLORS['gold'])

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🎲 Probabilidades (Monte Carlo)")
            fig_pie = px.pie(
                values=[mc_h, mc_d, mc_a], 
                names=[home_team, "Empate", away_team],
                color_discrete_sequence=[WC_COLORS['maroon'], WC_COLORS['gold'], WC_COLORS['navy']],
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            xp_h = (mc_h * 3) + (mc_d * 1)
            xp_a = (mc_a * 3) + (mc_d * 1)
            st.info(f"**xP:** {home_team}: {xp_h:.2f} | {away_team}: {xp_a:.2f}")

        with c2:
            st.markdown("### 🛡️ Radar de Força")
            categories = ['Ataque', 'Defesa', 'Elo', 'Tradição', 'Forma']
            def get_stats(team, elo, att, defen):
                s_att = np.clip(att.get(team, 1.0) * 40, 20, 100)
                s_def = np.clip((2.5 - defen.get(team, 1.0)) * 40, 20, 100)
                s_elo = np.clip((elo - 1000) / 10, 0, 100)
                return [s_att, s_def, s_elo, 75, 80]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=get_stats(home_team, res['elo_home'], attack, defense), theta=categories, fill='toself', name=home_team, line_color=WC_COLORS['maroon']))
            fig_radar.add_trace(go.Scatterpolar(r=get_stats(away_team, res['elo_away'], attack, defense), theta=categories, fill='toself', name=away_team, line_color=WC_COLORS['navy']))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
            st.plotly_chart(fig_radar, use_container_width=True)

with tab2:
    if 'res' in locals():
        st.markdown("### 🎯 Probabilidade de Placares Exatos")
        fig_heat = px.imshow(
            res['matrix'][:6, :6],
            labels=dict(x=f"Gols {away_team}", y=f"Gols {home_team}", color="Prob"),
            x=[str(i) for i in range(6)], y=[str(i) for i in range(6)],
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
        scores = [(f"{i}-{j}", res['matrix'][i,j]) for i in range(6) for j in range(6)]
        scores.sort(key=lambda x: x[1], reverse=True)
        st.table(pd.DataFrame(scores[:5], columns=["Placar", "Probabilidade"]).assign(Probabilidade=lambda x: x['Probabilidade'].map('{:.1%}'.format)))
    else:
        st.info("Execute a análise na primeira aba para ver os detalhes.")

with tab3:
    st.markdown("### 📈 Evolução Recente")
    h_h = elo_history[(elo_history['home_team'] == home_team) | (elo_history['away_team'] == home_team)].tail(30)
    h_a = elo_history[(elo_history['home_team'] == away_team) | (elo_history['away_team'] == away_team)].tail(30)
    
    def get_series(df_t, team):
        return [r['home_rating_after'] if r['home_team'] == team else r['away_rating_after'] for _, r in df_t.iterrows()]

    fig_evol = go.Figure()
    fig_evol.add_trace(go.Scatter(x=h_h['date'], y=get_series(h_h, home_team), name=home_team, line=dict(color=WC_COLORS['maroon'], width=3)))
    fig_evol.add_trace(go.Scatter(x=h_a['date'], y=get_series(h_a, away_team), name=away_team, line=dict(color=WC_COLORS['navy'], width=3)))
    fig_evol.update_layout(xaxis_title="Data", yaxis_title="Elo Rating")
    st.plotly_chart(fig_evol, use_container_width=True)

st.markdown("---")
st.caption("© 2026 World Cup Quant Intelligence | Elo & Poisson Models")
