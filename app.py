# app.py — Global Football Quant Intelligence v2.0 (Restaurado & Avançado)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

from utils.engine import (
    compute_elo_history, get_poisson_strengths, predict_match, monte_carlo,
    INITIAL_ELO, get_confederation, CONFEDERATIONS
)
from utils.advanced_metrics import (
    get_team_history, elo_volatility, elo_momentum, elo_efficiency,
    schedule_strength, golden_window, regression_speed, h2h_psychological_edge,
    home_advantage_by_confederation, kelly_criterion, team_consistency,
    xg_efficiency
)

# ===================== CONFIGURAÇÕES =====================
LOOKBACK_YEARS = 10
st.set_page_config(
    page_title="Global Football Quant Intelligence",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

# CSS custom dark theme
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #f0f0f0; }
    .sub-header { font-size: 1.1rem; color: #a0a0a0; margin-bottom: 1rem; }
    .metric-card { background: #1a1a2e; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #e94560; }
    .narrative-box { background: #16213e; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #0f3460; font-size: 0.95rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; }
    div[data-testid="stSidebarNav"] { background: #0f0f23; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Carregando base histórica...")
def load_data():
    df = pd.read_csv("data/results.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score'])
    name_map = {'Czechoslovakia': 'Czech Republic', 'Yugoslavia': 'Serbia',
                'Soviet Union': 'Russia', 'West Germany': 'Germany',
                'East Germany': 'Germany'}
    df['home_team'] = df['home_team'].replace(name_map)
    df['away_team'] = df['away_team'].replace(name_map)
    return df

@st.cache_data(show_spinner="Computando ratings Elo...")
def get_processed_data(df, lookback):
    elo_ratings, elo_history = compute_elo_history(df, lookback)
    attack, defense, league_avg = get_poisson_strengths(df, lookback)
    return elo_ratings, elo_history, attack, defense, league_avg

# ===================== LOAD =====================
df = load_data()
try:
    elo_ratings, elo_history, attack, defense, league_avg = get_processed_data(df, LOOKBACK_YEARS)
    teams = sorted(list(elo_ratings.keys()))
except Exception as e:
    st.error(f"Erro crítico no motor Elo: {e}")
    st.stop()

# ===================== SIDEBAR NAVIGATION =====================
st.sidebar.markdown("<div class='main-header'>⚽ Quant Football</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sub-header'>Navegação Inteligente</div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    [
        "🧠 Inteligência Elo",
        "🔮 Predição Quant",
        "⚔️ Confronto Histórico",
        "📊 Métricas Avançadas",
        "🏟️ Contexto de Campo",
        "🧬 Evolução Temporal",
        "🧮 Laboratório Quant"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()
home_team = st.sidebar.selectbox("Seleção Principal", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
away_team = st.sidebar.selectbox("Seleção Oponente", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)

if home_team == away_team:
    st.sidebar.warning("Selecione times diferentes.")
    st.stop()

neutral = st.sidebar.toggle("Campo Neutro", value=True)
lookback = st.sidebar.slider("Janela Temporal (anos)", 1, 20, LOOKBACK_YEARS)

if lookback != LOOKBACK_YEARS:
    elo_ratings, elo_history, attack, defense, league_avg = get_processed_data(df, lookback)

# ===================== FUNÇÕES AUXILIARES =====================
def narrate_elo_context(team, ratings, history):
    conf = get_confederation(team)
    rank = sorted(ratings.values(), reverse=True).index(ratings[team]) + 1
    vol = elo_volatility(history, team, window=10)
    mom = elo_momentum(history, team)
    eff_real, eff_edge = elo_efficiency(history, team, n_last=20)
    sched = schedule_strength(history, team, n_last=10)
    consist = team_consistency(history, team, n_last=20)

    narrative = f"""
    **{team}** ocupa a posição **#{rank}** global na Confederação **{conf}**.
    {"O time vive um momento de **ascensão** com momentum de +{:.1f} pts.".format(mom) if mom and mom > 15 else
     "O time está em **declínio** com momentum de {:.1f} pts.".format(mom) if mom and mom < -15 else
     "O time está em **estabilidade** de curto prazo."}
    {"Sua volatilidade de {:.1f} indica **instabilidade** — resultados imprevisíveis.".format(vol) if vol and vol > 25 else
     "Baixa volatilidade ({:.1f}) sugere **previsibilidade**.".format(vol) if vol else ""}
    {"Eficiência: conquistou {:.2f} pts/jogo vs. {:.2f} esperado — **superperformando** em {:.2f}.".format(eff_real, eff_real - eff_edge, -eff_edge) if eff_edge and eff_edge < 0 else
     "Eficiência: está **subperformando** em {:.2f} pts/jogo vs. expectativa.".format(eff_edge) if eff_edge else ""}
    """.strip()
    return narrative, vol, mom, eff_real, eff_edge, sched, consist

def get_team_elo_series(history_df, team):
    s = get_team_history(history_df, team)
    if s.empty:
        return pd.DataFrame()
    return s[['date', 'rating']].copy()

# ===================== PÁGINA 1: INTELIGÊNCIA ELO =====================
if page == "🧠 Inteligência Elo":
    st.markdown("<div class='main-header'>🧠 Inteligência Elo Global</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Ranking dinâmico com momentum, volatilidade e eficiência. Não é só um número — é uma história.</div>", unsafe_allow_html=True)

    ranking_data = []
    for team in teams:
        vol = elo_volatility(elo_history, team, window=10)
        mom = elo_momentum(elo_history, team)
        eff_real, eff_edge = elo_efficiency(elo_history, team, n_last=20)
        sched = schedule_strength(elo_history, team, n_last=10)
        consist = team_consistency(elo_history, team, n_last=20)
        conf = get_confederation(team)
        ranking_data.append({
            'Seleção': team, 'Elo': elo_ratings[team], 'Confederação': conf,
            'Volatilidade': vol, 'Momentum': mom, 'Eficiência Real': eff_real,
            'Edge Eficiência': -eff_edge if eff_edge is not np.nan else np.nan,
            'Força Agenda': sched, 'Consistência': consist
        })

    rank_df = pd.DataFrame(ranking_data).sort_values('Elo', ascending=False).reset_index(drop=True)
    rank_df.index += 1

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🏆 Ranking Quant Completo")
        display_df = rank_df[['Seleção', 'Elo', 'Confederação', 'Momentum', 'Volatilidade', 'Edge Eficiência']].head(50)
        st.dataframe(display_df.style.background_gradient(subset=['Elo'], cmap='RdYlGn')
                     .background_gradient(subset=['Momentum'], cmap='RdYlGn', vmin=-50, vmax=50)
                     .background_gradient(subset=['Volatilidade'], cmap='YlOrRd')
                     .format({'Elo': '{:.0f}', 'Momentum': '{:.1f}', 'Volatilidade': '{:.1f}', 'Edge Eficiência': '{:.2f}'}),
                     use_container_width=True, height=500)

    with col2:
        st.subheader("📜 Narrativa")
        for t in [home_team, away_team]:
            narr, vol, mom, eff_real, eff_edge, sched, consist = narrate_elo_context(t, elo_ratings, elo_history)
            st.markdown(f"<div class='narrative-box'>{narr}</div>", unsafe_allow_html=True)
            st.caption(f"Elo: {elo_ratings[t]:.0f} | Agenda: {sched:.0f if sched is not np.nan else 'N/A'} | Consistência: {consist:.3f if consist is not np.nan else 'N/A'}")
            st.divider()

    st.subheader("🌍 Distribuição de Força por Confederação")
    conf_stats = rank_df.groupby('Confederação').agg({'Elo': ['mean', 'std', 'max', 'count']}).reset_index()
    conf_stats.columns = ['Confederação', 'Média Elo', 'Desvio', 'Máximo', 'Seleções']
    fig_conf = px.scatter(conf_stats, x='Média Elo', y='Desvio', size='Seleções', color='Confederação',
                          text='Confederação', size_max=60,
                          title="Dispersão de Força: Média vs. Desigualdade Interna",
                          labels={'Desvio': 'Desigualdade Interna (σ Elo)', 'Média Elo': 'Força Média'})
    fig_conf.update_traces(textposition='top center')
    st.plotly_chart(fig_conf, use_container_width=True)

    st.subheader("🚀 Top 10 Momentum (Tendência de Curto Prazo)")
    top_mom = rank_df.dropna(subset=['Momentum']).sort_values('Momentum', ascending=False).head(10)
    fig_mom = px.bar(top_mom, x='Momentum', y='Seleção', orientation='h', color='Momentum',
                     color_continuous_scale='RdYlGn', title="Times em Ascensão Imediata")
    st.plotly_chart(fig_mom, use_container_width=True)

# ===================== PÁGINA 2: PREDIÇÃO QUANT =====================
elif page == "🔮 Predição Quant":
    st.markdown("<div class='main-header'>🔮 Predição Quant Avançada</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Modelo Poisson-Dixon-Coles + Monte Carlo. Probabilidades, xG, xP e Goal Supremacy.</div>", unsafe_allow_html=True)

    res = predict_match(home_team, away_team, neutral, attack, defense, league_avg, elo_ratings, rho=-0.08)
    mc_h, mc_d, mc_a = monte_carlo(res['l_home'], res['l_away'], iterations=20000)

    elo_diff = res['elo_home'] - res['elo_away']
    supremacy = res['goal_supremacy']
    narrative_pred = f"""
    **{home_team}** ({res['elo_home']:.0f} Elo) vs. **{away_team}** ({res['elo_away']:.0f} Elo).
    Diferença de rating: **{elo_diff:+.0f}**.
    Goal Supremacy projetado: **{supremacy:+.2f}** (expectativa de diferença de gols).
    {"O modelo detecta **vantagem significativa** para a casa." if supremacy > 0.8 else
     "Confronto **equilibrado** — o empate é o cenário mais provável." if abs(supremacy) < 0.3 else
     "O modelo favorece **levemente** o visitante." if supremacy < -0.3 else
     "O modelo favorece **levemente** a casa."}
    """.strip()
    st.markdown(f"<div class='narrative-box'>{narrative_pred}</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Elo Casa", f"{res['elo_home']:.0f}", f"{elo_diff:+.0f}")
    c2.metric("Elo Fora", f"{res['elo_away']:.0f}")
    c3.metric("xG Casa", f"{res['xg_home']:.2f}")
    c4.metric("xG Fora", f"{res['xg_away']:.2f}")
    c5.metric("Goal Supremacy", f"{supremacy:+.2f}")

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("🎲 Probabilidades (Monte Carlo 20k)")
        probs = [mc_h, mc_d, mc_a]
        labels = [home_team, "Empate", away_team]
        colors = ['#e94560', '#f0f0f0', '#0f3460']
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=probs, hole=0.5,
                                          marker_colors=colors, textinfo='label+percent',
                                          insidetextorientation='radial')])
        fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

        xp_h = (mc_h * 3) + (mc_d * 1)
        xp_a = (mc_a * 3) + (mc_d * 1)
        st.info(f"**xP (Pontos Esperados):** {home_team}: {xp_h:.2f} | {away_team}: {xp_a:.2f}")

    with col_right:
        st.subheader("🎯 Matriz de Placares (Dixon-Coles)")
        fig_heat = px.imshow(
            res['matrix'][:7, :7],
            labels=dict(x=f"Gols {away_team}", y=f"Gols {home_team}", color="Prob"),
            x=[str(i) for i in range(7)], y=[str(i) for i in range(7)],
            color_continuous_scale="Blues", aspect="equal"
        )
        fig_heat.update_traces(text=np.round(res['matrix'][:7, :7], 3),
                               texttemplate="%{text:.1%}", textfont={"size": 10})
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        scores = [(f"{i}-{j}", res['matrix'][i, j]) for i in range(8) for j in range(8)]
        scores.sort(key=lambda x: x[1], reverse=True)
        st.markdown("**Top 5 Placares Prováveis**")
        st.table(pd.DataFrame(scores[:5], columns=["Placar", "Prob"]).assign(Prob=lambda x: x['Prob'].map('{:.1%}'.format)))

    with c_m2:
        st.markdown("**Mercados de Gols & BTTS**")
        prob_over_15 = np.sum([res['matrix'][i, j] for i in range(11) for j in range(11) if i + j > 1.5])
        prob_over_25 = np.sum([res['matrix'][i, j] for i in range(11) for j in range(11) if i + j > 2.5])
        prob_over_35 = np.sum([res['matrix'][i, j] for i in range(11) for j in range(11) if i + j > 3.5])
        prob_btts = np.sum([res['matrix'][i, j] for i in range(1, 11) for j in range(1, 11)])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Over 1.5", f"{prob_over_15:.1%}")
        m2.metric("Over 2.5", f"{prob_over_25:.1%}")
        m3.metric("Over 3.5", f"{prob_over_35:.1%}")
        m4.metric("BTTS", f"{prob_btts:.1%}")

    st.subheader("🛡️ Radar de Força Quant")
    categories = ['Ataque', 'Defesa', 'Elo', 'Momentum', 'Consistência', 'Eficiência']

    def get_radar_stats(team, elo, att, defen, hist):
        s_att = np.clip(att.get(team, 1.0) * 50, 15, 100)
        s_def = np.clip((2.5 - defen.get(team, 1.0)) * 40, 15, 100)
        s_elo = np.clip((elo - 1000) / 12, 0, 100)
        mom = elo_momentum(hist, team, short_window=5, long_window=15)
        s_mom = np.clip(50 + (mom if mom is not np.nan else 0), 0, 100)
        consist = team_consistency(hist, team, n_last=20)
        s_cons = np.clip(100 - (consist * 1000 if consist is not np.nan else 50), 0, 100)
        eff_real, eff_edge = elo_efficiency(hist, team, n_last=20)
        s_eff = np.clip(50 + (-eff_edge * 30 if eff_edge is not np.nan else 0), 0, 100)
        return [s_att, s_def, s_elo, s_mom, s_cons, s_eff]

    fig_radar = go.Figure()
    stats_h = get_radar_stats(home_team, res['elo_home'], attack, defense, elo_history)
    stats_a = get_radar_stats(away_team, res['elo_away'], attack, defense, elo_history)
    fig_radar.add_trace(go.Scatterpolar(r=stats_h + [stats_h[0]], theta=categories + [categories[0]],
                                         fill='toself', name=home_team, line_color='#e94560'))
    fig_radar.add_trace(go.Scatterpolar(r=stats_a + [stats_a[0]], theta=categories + [categories[0]],
                                         fill='toself', name=away_team, line_color='#0f3460'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

# ===================== PÁGINA 3: CONFRONTO HISTÓRICO =====================
elif page == "⚔️ Confronto Histórico":
    st.markdown("<div class='main-header'>⚔️ Confronto Histórico Profundo</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>H2H, psychological edge, tendências de confronto e domínio psicológico.</div>", unsafe_allow_html=True)

    h2h_data = h2h_psychological_edge(df, home_team, away_team, lookback_years=30)

    if h2h_data['total'] > 0:
        edge_narr = f"""
        Nos últimos **{h2h_data['total']}** confrontos, **{home_team}** venceu **{h2h_data['wins_a']}**,
        **{away_team}** venceu **{h2h_data['wins_b']}**, com **{h2h_data['draws']}** empates.
        Saldo de gols: **{h2h_data['gd_sum']:+d}** para {home_team}.
        {"**Domínio psicológico ativo**: " + str(h2h_data['streak']) + " vitórias consecutivas." if h2h_data['streak'] >= 2 else
         "Sem sequência de domínio atual."}
        Classificação de edge: **{h2h_data['edge']}** (score: {h2h_data['dominance']:.2f}).
        """.strip()
    else:
        edge_narr = "Sem histórico de confrontos diretos nos últimos 30 anos."
    st.markdown(f"<div class='narrative-box'>{edge_narr}</div>", unsafe_allow_html=True)

    cutoff = df['date'].max() - timedelta(days=365 * 30)
    h2h_df = df[((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
                ((df['home_team'] == away_team) & (df['away_team'] == home_team))]
    h2h_df = h2h_df[h2h_df['date'] >= cutoff].sort_values('date')

    if not h2h_df.empty:
        fig_timeline = go.Figure()
        for _, row in h2h_df.iterrows():
            if row['home_team'] == home_team:
                gd = row['home_score'] - row['away_score']
                color = '#e94560' if gd > 0 else '#0f3460' if gd < 0 else '#f0f0f0'
            else:
                gd = row['away_score'] - row['home_score']
                color = '#e94560' if gd > 0 else '#0f3460' if gd < 0 else '#f0f0f0'
            fig_timeline.add_trace(go.Scatter(
                x=[row['date']], y=[gd],
                mode='markers', marker=dict(size=12, color=color),
                showlegend=False, hovertext=f"{row['home_team']} {row['home_score']} x {row['away_score']} {row['away_team']}<br>{row['tournament']}<br>{row['date'].strftime('%Y-%m-%d')}"
            ))
        fig_timeline.update_layout(title="Timeline de Confrontos (Diferença de Gols)",
                                    yaxis_title=f"GD favorável a {home_team}", xaxis_title="Ano",
                                    hovermode="closest")
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.subheader("📊 Estatísticas do Confronto")
        h2h_df['gd_for_home'] = h2h_df.apply(
            lambda r: r['home_score'] - r['away_score'] if r['home_team'] == home_team else r['away_score'] - r['home_score'], axis=1)
        avg_gd = h2h_df['gd_for_home'].mean()
        avg_goals = (h2h_df['home_score'] + h2h_df['away_score']).mean()
        btts_rate = ((h2h_df['home_score'] > 0) & (h2h_df['away_score'] > 0)).mean()
        over25 = ((h2h_df['home_score'] + h2h_df['away_score']) > 2.5).mean()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Média GD", f"{avg_gd:+.2f}")
        k2.metric("Média Gols", f"{avg_goals:.2f}")
        k3.metric("BTTS %", f"{btts_rate:.1%}")
        k4.metric("Over 2.5 %", f"{over25:.1%}")
        k5.metric("Jogos", len(h2h_df))

        results_map = []
        for _, row in h2h_df.iterrows():
            if row['home_team'] == home_team:
                if row['home_score'] > row['away_score']: results_map.append('Vitória Casa')
                elif row['home_score'] == row['away_score']: results_map.append('Empate')
                else: results_map.append('Vitória Fora')
            else:
                if row['away_score'] > row['home_score']: results_map.append('Vitória Casa')
                elif row['away_score'] == row['home_score']: results_map.append('Empate')
                else: results_map.append('Vitória Fora')
        res_counts = pd.Series(results_map).value_counts().reset_index()
        res_counts.columns = ['Resultado', 'Contagem']
        fig_res = px.pie(res_counts, values='Contagem', names='Resultado', hole=0.4,
                         color_discrete_map={'Vitória Casa': '#e94560', 'Empate': '#f0f0f0', 'Vitória Fora': '#0f3460'})
        st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("Nenhum confronto direto encontrado no período.")

# ===================== PÁGINA 4: MÉTRICAS AVANÇADAS =====================
elif page == "📊 Métricas Avançadas":
    st.markdown("<div class='main-header'>📊 Métricas Avançadas</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Volatilidade, eficiência, consistência, força de agenda e janelas de ouro. Métricas que os apps comuns ignoram.</div>", unsafe_allow_html=True)

    for team in [home_team, away_team]:
        st.divider()
        st.subheader(f"🔬 {team}")
        vol = elo_volatility(elo_history, team, window=10)
        mom = elo_momentum(elo_history, team)
        eff_real, eff_edge = elo_efficiency(elo_history, team, n_last=20)
        sched = schedule_strength(elo_history, team, n_last=10)
        consist = team_consistency(elo_history, team, n_last=20)
        gw_start, gw_end = golden_window(elo_history, team, min_window=8, threshold_pct=0.95)
        reg_speed = regression_speed(elo_history, team)
        xg_eff, avg_g = xg_efficiency(df, team, lookback_years=5)

        cols = st.columns(4)
        cols[0].metric("Volatilidade Elo", f"{vol:.1f}" if vol is not np.nan else "N/A",
                       help="Quanto maior, mais imprevisível")
        cols[1].metric("Momentum", f"{mom:+.1f}" if mom is not np.nan else "N/A",
                       help="Diferença média curta vs. longa prazo")
        cols[2].metric("Eficiência", f"{eff_real:.2f} pts/j" if eff_real is not np.nan else "N/A",
                       help="Pontos reais por jogo")
        cols[3].metric("Consistência (CV)", f"{consist:.4f}" if consist is not np.nan else "N/A",
                       help="Coeficiente de variação do rating. Menor = mais consistente")

        cols2 = st.columns(4)
        cols2[0].metric("Força Agenda", f"{sched:.0f}" if sched is not np.nan else "N/A",
                        help="Elo médio dos adversários recentes")
        cols2[1].metric("Velocidade Regressão", f"{reg_speed:.0f} jgs" if reg_speed is not np.nan else "N/A",
                        help="Jogos para regressar à média após pico")
        cols2[2].metric("xG Efficiency", f"{xg_eff:.2f}" if xg_eff is not np.nan else "N/A",
                        help="Gols reais / gols esperados")
        cols2[3].metric("Média Gols", f"{avg_g:.2f}" if avg_g is not np.nan else "N/A")

        if gw_start and gw_end:
            st.success(f"🏆 **Janela de Ouro detectada**: {gw_start.strftime('%Y-%m')} → {gw_end.strftime('%Y-%m')}")
        else:
            st.info("Nenhuma janela de ouro clara detectada (requer mínimo 8 jogos).")

    st.divider()
    st.subheader("📈 Comparativo Visual de Métricas")
    metrics_comp = []
    for team in teams:
        vol = elo_volatility(elo_history, team, window=10)
        mom = elo_momentum(elo_history, team)
        eff_real, eff_edge = elo_efficiency(elo_history, team, n_last=20)
        consist = team_consistency(elo_history, team, n_last=20)
        metrics_comp.append({
            'Seleção': team, 'Elo': elo_ratings[team], 'Volatilidade': vol,
            'Momentum': mom, 'Eficiência': eff_real, 'Consistência': consist
        })
    mdf = pd.DataFrame(metrics_comp).dropna()

    fig_scatter = px.scatter(mdf, x='Elo', y='Momentum', size='Eficiência', color='Volatilidade',
                             hover_data=['Seleção', 'Consistência'],
                             title="Mapa Quant: Elo vs. Momentum (tamanho = eficiência)",
                             color_continuous_scale='YlOrRd')
    fig_scatter.add_vline(x=elo_ratings[home_team], line_dash="dash", line_color="#e94560", annotation_text=home_team)
    fig_scatter.add_vline(x=elo_ratings[away_team], line_dash="dash", line_color="#0f3460", annotation_text=away_team)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ===================== PÁGINA 5: CONTEXTO DE CAMPO =====================
elif page == "🏟️ Contexto de Campo":
    st.markdown("<div class='main-header'>🏟️ Contexto de Campo</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Home Advantage por confederação, fatores contextuais e análise de neutralidade.</div>", unsafe_allow_html=True)

    ha_df = home_advantage_by_confederation(df, elo_history, CONFEDERATIONS)
    if not ha_df.empty:
        st.subheader("🌍 Home Advantage Real por Confederação")
        fig_ha = px.bar(ha_df.sort_values('Advantage Index', ascending=True),
                        x='Advantage Index', y='Confederação', orientation='h',
                        color='Advantage Index', color_continuous_scale='RdYlGn',
                        text='Advantage Index',
                        labels={'Advantage Index': 'Índice de Vantagem (0 = neutro)'})
        fig_ha.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        st.plotly_chart(fig_ha, use_container_width=True)

        st.markdown("""
        **Interpretação:** A CONMEBOL e CAF apresentam os maiores índices de home advantage, 
        provavelmente devido a distâncias de viagem, altitude e atmosfera hostil. 
        A UEFA tem o índice mais baixo, refletindo infraestrutura equilibrada e menor variação geográfica intra-confederação.
        """)

    st.divider()
    st.subheader("⚖️ Análise Contextual do Confronto Selecionado")
    conf_h = get_confederation(home_team)
    conf_a = get_confederation(away_team)

    context_narr = f"""
    **{home_team}** ({conf_h}) vs. **{away_team}** ({conf_a}).
    {"Campo neutro ativado — home advantage neutralizado." if neutral else
     f"Jogo em casa para {home_team}. Na {conf_h}, a vantagem de casa é significativa."}
    {"Confronto inter-confederação — adaptação tática e fuso horário podem ser fatores." if conf_h != conf_a else
     "Confronto intra-confederação — familiaridade tática elevada."}
    """.strip()
    st.markdown(f"<div class='narrative-box'>{context_narr}</div>", unsafe_allow_html=True)

    if not neutral:
        st.subheader("📉 Impacto da Neutralidade (Simulação)")
        res_neutral = predict_match(home_team, away_team, True, attack, defense, league_avg, elo_ratings, rho=-0.08)
        res_home = predict_match(home_team, away_team, False, attack, defense, league_avg, elo_ratings, rho=-0.08)

        delta_h = res_home['p_home'] - res_neutral['p_home']
        delta_d = res_home['p_draw'] - res_neutral['p_draw']
        delta_a = res_home['p_away'] - res_neutral['p_away']

        d1, d2, d3 = st.columns(3)
        d1.metric(f"Δ Prob {home_team}", f"{delta_h:+.1%}", help="Ganho/perda de probabilidade com campo")
        d2.metric("Δ Prob Empate", f"{delta_d:+.1%}")
        d3.metric(f"Δ Prob {away_team}", f"{delta_a:+.1%}")

        fig_delta = go.Figure(go.Waterfall(
            name="Impacto", orientation="v",
            measure=["relative", "relative", "relative"],
            x=[home_team, "Empate", away_team],
            y=[delta_h * 100, delta_d * 100, delta_a * 100],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#0f3460"}},
            increasing={"marker": {"color": "#e94560"}}
        ))
        fig_delta.update_layout(title="Impacto da Vantagem de Campo (pp)", yaxis_title="Δ Probabilidade (%)")
        st.plotly_chart(fig_delta, use_container_width=True)

# ===================== PÁGINA 6: EVOLUÇÃO TEMPORAL =====================
elif page == "🧬 Evolução Temporal":
    st.markdown("<div class='main-header'>🧬 Evolução Temporal</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Séries temporais de Elo com decomposição, ciclos de geração e detecção de janelas de ouro.</div>", unsafe_allow_html=True)

    for team in [home_team, away_team]:
        st.divider()
        st.subheader(f"📈 {team}")
        s = get_team_history(elo_history, team, n_last=200)
        if s.empty or len(s) < 5:
            st.warning(f"Dados insuficientes para {team}.")
            continue

        fig_evol = go.Figure()
        fig_evol.add_trace(go.Scatter(x=s['date'], y=s['rating'], mode='lines',
                                       name='Elo', line=dict(color='#e94560', width=2)))
        s['ma_10'] = s['rating'].rolling(window=10, min_periods=1, center=True).mean()
        fig_evol.add_trace(go.Scatter(x=s['date'], y=s['ma_10'], mode='lines',
                                       name='MA-10', line=dict(color='#0f3460', width=2, dash='dash')))
        s['std_10'] = s['rating'].rolling(window=10, min_periods=1, center=True).std()
        fig_evol.add_trace(go.Scatter(x=s['date'], y=s['ma_10'] + s['std_10'],
                                       fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                                       showlegend=False, hoverinfo='skip'))
        fig_evol.add_trace(go.Scatter(x=s['date'], y=s['ma_10'] - s['std_10'],
                                       fill='tonexty', fillcolor='rgba(15,52,96,0.2)',
                                       mode='lines', line_color='rgba(0,0,0,0)',
                                       showlegend=False, hoverinfo='skip'))
        fig_evol.update_layout(title=f"Evolução de Elo — {team}", xaxis_title="Data", yaxis_title="Rating",
                               hovermode="x unified")
        st.plotly_chart(fig_evol, use_container_width=True)

        from scipy.signal import find_peaks
        ratings_arr = s['rating'].values
        peaks, _ = find_peaks(ratings_arr, distance=10, prominence=15)
        valleys, _ = find_peaks(-ratings_arr, distance=10, prominence=15)

        if len(peaks) > 0 or len(valleys) > 0:
            st.caption(f"**Ciclos detectados**: {len(peaks)} picos, {len(valleys)} vales. Cada ciclo representa uma geração ou mudança tática.")
        else:
            st.caption("Nenhum ciclo claro detectado — evolução relativamente linear.")

        gw_start, gw_end = golden_window(elo_history, team, min_window=8, threshold_pct=0.95)
        if gw_start and gw_end:
            st.success(f"🏆 Janela de Ouro: {gw_start.strftime('%Y-%m-%d')} → {gw_end.strftime('%Y-%m-%d')}")
        else:
            st.info("Sem janela de ouro clara no período.")

    st.divider()
    st.subheader("🔄 Comparativo Temporal Cruzado")
    s_h = get_team_history(elo_history, home_team, n_last=200)
    s_a = get_team_history(elo_history, away_team, n_last=200)
    if not s_h.empty and not s_a.empty:
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=s_h['date'], y=s_h['rating'], mode='lines',
                                       name=home_team, line=dict(color='#e94560', width=2)))
        fig_comp.add_trace(go.Scatter(x=s_a['date'], y=s_a['rating'], mode='lines',
                                       name=away_team, line=dict(color='#0f3460', width=2)))
        fig_comp.update_layout(title="Elo Cruzado — Evolução Comparada", xaxis_title="Data", yaxis_title="Rating",
                               hovermode="x unified")
        st.plotly_chart(fig_comp, use_container_width=True)

# ===================== PÁGINA 7: LABORATÓRIO QUANT =====================
elif page == "🧮 Laboratório Quant":
    st.markdown("<div class='main-header'>🧮 Laboratório Quant</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Backtesting, Kelly Criterion, edge detection e simulação de portfólio. Para quem pensa como um fundo quant.</div>", unsafe_allow_html=True)

    res = predict_match(home_team, away_team, neutral, attack, defense, league_avg, elo_ratings, rho=-0.08)
    mc_h, mc_d, mc_a = monte_carlo(res['l_home'], res['l_away'], iterations=20000)

    st.subheader("💰 Kelly Criterion & Edge Detection")
    st.markdown("Insira as odds do mercado para detectar edge e calcular stake ótima (Kelly fractional 25%).")

    col_odds = st.columns(3)
    odds_h = col_odds[0].number_input(f"Odd {home_team}", min_value=1.01, value=2.50, step=0.05, format="%.2f")
    odds_d = col_odds[1].number_input("Odd Empate", min_value=1.01, value=3.20, step=0.05, format="%.2f")
    odds_a = col_odds[2].number_input(f"Odd {away_team}", min_value=1.01, value=3.10, step=0.05, format="%.2f")

    fair_h = 1 / mc_h
    fair_d = 1 / mc_d
    fair_a = 1 / mc_a

    edge_h = (mc_h * odds_h - 1) if odds_h > 0 else -1
    edge_d = (mc_d * odds_d - 1) if odds_d > 0 else -1
    edge_a = (mc_a * odds_a - 1) if odds_a > 0 else -1

    kelly_h = kelly_criterion(mc_h, odds_h, fraction=0.25)
    kelly_d = kelly_criterion(mc_d, odds_d, fraction=0.25)
    kelly_a = kelly_criterion(mc_a, odds_a, fraction=0.25)

    st.divider()
    st.markdown("**Fair Odds (Modelo):** {:.2f} | {:.2f} | {:.2f}".format(fair_h, fair_d, fair_a))

    e1, e2, e3 = st.columns(3)
    e1.metric(f"Edge {home_team}", f"{edge_h:+.2%}", f"Kelly: {kelly_h:.2%}",
              delta_color="normal" if edge_h > 0 else "inverse")
    e2.metric("Edge Empate", f"{edge_d:+.2%}", f"Kelly: {kelly_d:.2%}",
              delta_color="normal" if edge_d > 0 else "inverse")
    e3.metric(f"Edge {away_team}", f"{edge_a:+.2%}", f"Kelly: {kelly_a:.2%}",
              delta_color="normal" if edge_a > 0 else "inverse")

    best_edge = max((edge_h, home_team, kelly_h), (edge_d, "Empate", kelly_d), (edge_a, away_team, kelly_a))
    if best_edge[0] > 0:
        st.success(f"🎯 **Edge detectado em {best_edge[1]}**: +{best_edge[0]:.2%}. Stake ótima (Kelly 25%): {best_edge[2]:.2%} do bankroll.")
    else:
        st.warning("⚠️ Sem edge detectado nas odds inseridas. O mercado está eficiente para este confronto.")

    st.divider()
    st.subheader("📊 Simulação de Portfólio (Backtesting Simplificado)")
    st.markdown("Simulação: se você apostasse em todos os jogos do time nos últimos 20 jogos com edge > 5%.")

    for team in [home_team, away_team]:
        s = get_team_history(elo_history, team).tail(20)
        if s.empty:
            continue

        bankroll = 1000
        stakes = []
        for _, row in s.iterrows():
            prob_win = row['expected']
            assumed_odd = 1.8
            edge = prob_win * assumed_odd - 1
            if edge > 0.05:
                kelly = kelly_criterion(prob_win, assumed_odd, fraction=0.25)
                stake = bankroll * kelly
                result = row['result']
                profit = stake * (assumed_odd - 1) if result == 1 else -stake if result == 0 else -stake * 0.5
                bankroll += profit
                stakes.append({'stake': stake, 'profit': profit, 'bankroll': bankroll, 'edge': edge})

        if stakes:
            port_df = pd.DataFrame(stakes)
            fig_port = go.Figure()
            fig_port.add_trace(go.Scatter(y=port_df['bankroll'], mode='lines+markers',
                                           name=f"Bankroll {team}", line=dict(width=2)))
            fig_port.add_hline(y=1000, line_dash="dash", line_color="red")
            fig_port.update_layout(title=f"Evolução do Bankroll — {team}", yaxis_title="Bankroll ($)", xaxis_title="Aposta #")
            st.plotly_chart(fig_port, use_container_width=True)
            roi = (port_df['bankroll'].iloc[-1] - 1000) / 1000
            st.caption(f"ROI simulado: {roi:+.2%} | Apostas: {len(port_df)} | Edge médio: {port_df['edge'].mean():.2%}")
        else:
            st.info(f"Nenhuma oportunidade com edge > 5% encontrada para {team} nos últimos 20 jogos.")

st.divider()
st.caption("© 2026 Global Football Quant Intelligence | Motor Elo-Poisson-Dixon-Coles | Dados históricos internacionais.")
