"""
Global Football Intelligence
Painel quantitativo: Elo Ratings, Head-to-Head, Predição Poisson
Dataset: International football results (1872-2024) + Transfermarkt + Estádios (Wikidata)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ⚡ Para o mapa interativo real (OpenStreetMap)
import folium
from folium.plugins import MarkerCluster, Fullscreen
from streamlit_folium import st_folium

from elo_engine import (
    load_former_names, build_name_map, normalize_team_names,
    compute_elo_history, get_elo_timeseries, predict_match,
    team_attack_defense_strength, poisson_match_probs, INITIAL_ELO
)
from club_engine import (
    load_club_data, valuation_age_curve, top_transfers,
    transfer_flow_by_league, nationality_distribution, club_summary, COMPETITION_LABELS
)

# ---------------------------------------------------------------------------
# CONFIG & THEME
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Global Football Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#d4af37"
ACCENT2 = "#1f77b4"
POSITIVE = "#2ca02c"
NEGATIVE = "#d62728"

# Template claro e minimalista (Clean White)
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="IBM Plex Mono, monospace", size=12, color="#111111"),
        colorway=[ACCENT, ACCENT2, "#f72585", "#4ade80", "#fb923c", "#a78bfa"],
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(255,255,255,0.9)"),
    )
)

st.markdown(f"""
<style>
/* Otimização de margens do layout */
.block-container {{ padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem; }}
[data-testid="stMetric"] {{ padding-bottom: 0px; }}

h1, h2, h3 {{ font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.5px; }}
h1 {{ color: {ACCENT} !important; border-bottom: 2px solid {ACCENT}; padding-bottom: 12px; }}
[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem; }}
.stTabs [aria-selected="true"] {{ color: {ACCENT} !important; border-bottom-color: {ACCENT} !important; }}
.caption-box {{
    border-left: 3px solid {ACCENT};
    padding: 0.6rem 1rem; font-size: 0.85rem; opacity: 0.85;
    font-family: 'IBM Plex Mono', monospace; margin-bottom: 1rem;
}}
.tier-badge {{
    display: inline-block; padding: 2px 10px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 600;
    border: 1px solid {ACCENT}; color: {ACCENT};
}}
.form-pill {{
    display: inline-block; width: 22px; height: 22px; line-height: 22px;
    text-align: center; border-radius: 50%; font-size: 0.7rem; font-weight: 700;
    font-family: 'IBM Plex Mono', monospace; color: white; margin-right: 3px;
}}
.form-w {{ background-color: {POSITIVE}; }}
.form-d {{ background-color: #888; }}
.form-l {{ background-color: {NEGATIVE}; }}
</style>
""", unsafe_allow_html=True)


def elo_tier(elo):
    if elo >= 1900:
        return "ELITE", ACCENT
    elif elo >= 1750:
        return "TOP CONTENDER", ACCENT2
    elif elo >= 1600:
        return "COMPETITIVE", POSITIVE
    elif elo >= 1450:
        return "DEVELOPING", "#888"
    else:
        return "EMERGING", NEGATIVE


def form_pills_html(results_df, team, n=5):
    sub = results_df[(results_df["home_team"] == team) | (results_df["away_team"] == team)].sort_values("date", ascending=False).head(n)
    html = ""
    for _, r in sub.iterrows():
        is_home = r["home_team"] == team
        gf = r["home_score"] if is_home else r["away_score"]
        ga = r["away_score"] if is_home else r["home_score"]
        if gf > ga:
            cls, letter = "w", "V"
        elif gf < ga:
            cls, letter = "l", "D"
        else:
            cls, letter = "d", "E"   # 🔧 CORREÇÃO AQUI
        html += f'<span class="form-pill form-{cls}">{letter}</span>'
    return html if html else "<i>sem jogos recentes</i>"


# ---------------------------------------------------------------------------
# DATA LOADING (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando base histórica e calculando Elo Ratings...")
def load_all_data():
    data_dir = Path(__file__).parent / "data"
    results = pd.read_csv(data_dir / "results.csv")
    goalscorers = pd.read_csv(data_dir / "goalscorers.csv")
    shootouts = pd.read_csv(data_dir / "shootouts.csv")
    former_names = load_former_names()

    name_map = build_name_map(former_names)
    history, current_ratings, enriched = compute_elo_history(results, name_map)

    results_norm = normalize_team_names(results, name_map)
    results_norm["date"] = pd.to_datetime(results_norm["date"])

    goalscorers_norm = normalize_team_names(goalscorers, name_map, cols=("home_team", "away_team", "team"))
    goalscorers_norm["date"] = pd.to_datetime(goalscorers_norm["date"])

    shootouts_norm = normalize_team_names(shootouts, name_map, cols=("home_team", "away_team", "winner"))

    strength, avg_h, avg_a = team_attack_defense_strength(results, name_map, lookback_years=10)

    return {
        "results": results_norm,
        "goalscorers": goalscorers_norm,
        "shootouts": shootouts_norm,
        "history": history,
        "current_ratings": current_ratings,
        "enriched": enriched,
        "strength": strength,
        "avg_home": avg_h,
        "avg_away": avg_a,
    }


DATA = load_all_data()


@st.cache_data(show_spinner="Carregando dados de mercado (Transfermarkt)...")
def load_club_dataset():
    try:
        return load_club_data()
    except FileNotFoundError:
        st.warning("Diretório 'data_clubs' não encontrado. Módulo de clubes indisponível.")
        return None


CLUB_DATA = load_club_dataset()

RESULTS = DATA["results"]
HISTORY = DATA["history"]
CURRENT_RATINGS = DATA["current_ratings"]
ENRICHED = DATA["enriched"]
STRENGTH = DATA["strength"]

ALL_TEAMS = sorted(CURRENT_RATINGS.keys())


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("⚽ Global Football Intelligence")
st.markdown(
    f'<div class="caption-box">'
    f'Elo Rating Engine &middot; Head-to-Head Analytics &middot; Modelo Poisson de Predição &nbsp;|&nbsp; '
    f'Base histórica: {RESULTS["date"].min().date()} — {RESULTS["date"].max().date()} '
    f'&nbsp;|&nbsp; {len(RESULTS):,} partidas</div>'.replace(",", "."),
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Elo Ranking & Evolução",
    "🆚 Head-to-Head",
    "🎯 Predição Poisson",
    "🏆 Tournament Analytics",
    "💰 Player Market Intelligence",
    "ℹ️ Metodologia",
    "🗺️ Estádios"
])


# ---------------------------------------------------------------------------
# TAB 1 — ELO RANKING & EVOLUÇÃO (REORGANIZADO)
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("📊 Ranking e Métricas Avançadas")

    # Preparar dados de ranking com métricas adicionais
    cutoff_12m = HISTORY["date"].max() - pd.DateOffset(months=12)
    recent_12m = HISTORY[HISTORY["date"] >= cutoff_12m]

    ranking_data = []
    for team in ALL_TEAMS:
        elo_atual = CURRENT_RATINGS[team]
        ts = HISTORY[HISTORY["team"] == team].sort_values("date")
        if len(ts) >= 2:
            ts_before = ts[ts["date"] <= cutoff_12m]
            elo_12m = ts_before["elo"].iloc[-1] if not ts_before.empty else elo_atual
        else:
            elo_12m = elo_atual
        delta_12m = elo_atual - elo_12m
        team_strength = STRENGTH.get(team, {})
        if isinstance(team_strength, dict):
            atk = team_strength.get("attack", 1.0)
            defense = team_strength.get("defense", 1.0)
        else:
            atk, defense = 1.0, 1.0
        jogos_12m = len(recent_12m[recent_12m["team"] == team])
        ranking_data.append({
            "Seleção": team,
            "Elo": round(elo_atual, 1),
            "Δ12m": round(delta_12m, 1),
            "Ataque": round(atk, 2),
            "Defesa": round(defense, 2),
            "Jogos (12m)": jogos_12m,
            "Tier": elo_tier(elo_atual)[0]
        })

    ranking_df = pd.DataFrame(ranking_data)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values("Elo", ascending=False).reset_index(drop=True)
        ranking_df["Rank"] = range(1, len(ranking_df) + 1)
        ranking_df["Δ12m"] = ranking_df["Δ12m"].fillna(0)
        ranking_df["PosAnterior"] = ranking_df["Rank"] + (ranking_df["Δ12m"] / 5).round().astype(int)
        ranking_df["ΔPos"] = ranking_df["PosAnterior"] - ranking_df["Rank"]
        ranking_df["ΔPos"] = ranking_df["ΔPos"].clip(-20, 20)
        cols_order = ["Rank", "Seleção", "Elo", "Δ12m", "ΔPos", "Ataque", "Defesa", "Jogos (12m)", "Tier"]
        ranking_df = ranking_df[cols_order]
    else:
        ranking_df = pd.DataFrame(columns=["Rank", "Seleção", "Elo", "Δ12m", "ΔPos", "Ataque", "Defesa", "Jogos (12m)", "Tier"])

    # --- Layout com duas colunas: Ranking + Simulador ---
    col_rank, col_sim = st.columns([1.4, 1], gap="large")
    with col_rank:
        st.markdown("#### 🏆 Ranking Elo Atual")
        top_n = st.slider("Top N seleções", 5, 50, 20, key="top_n_elo")
        search_team = st.text_input("🔍 Buscar seleção", placeholder="Digite o nome...")
        df_display = ranking_df.copy()
        if search_team:
            df_display = df_display[df_display["Seleção"].str.contains(search_team, case=False)]
        st.dataframe(
            df_display.head(top_n) if not search_team else df_display,
            use_container_width=True,
            height=520,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                "Elo": st.column_config.ProgressColumn("Elo", min_value=1300, max_value=2100, format="%.0f"),
                "Δ12m": st.column_config.Column("Δ12m", help="Variação nos últimos 12 meses"),
                "ΔPos": st.column_config.Column("Δ Pos.", help="Variação aproximada de posição (estimada)"),
                "Ataque": st.column_config.NumberColumn("Força Ofensiva", format="%.2f"),
                "Defesa": st.column_config.NumberColumn("Força Defensiva", format="%.2f"),
                "Jogos (12m)": st.column_config.Column("Jogos", help="Partidas nos últimos 12 meses"),
            }
        )

    with col_sim:
        with st.container(border=True):
            st.markdown("#### 📈 Simulador de Confronto (Elo)")
            sim_home = st.selectbox("Seleção da casa", ALL_TEAMS, index=ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 0, key="sim_home")
            sim_away = st.selectbox("Seleção visitante", ALL_TEAMS, index=ALL_TEAMS.index("Argentina") if "Argentina" in ALL_TEAMS else 1, key="sim_away")
            neutral_sim = st.checkbox("Campo neutro", value=True, key="neutral_sim")
            if sim_home != sim_away:
                elo_h = CURRENT_RATINGS.get(sim_home, INITIAL_ELO)
                elo_a = CURRENT_RATINGS.get(sim_away, INITIAL_ELO)
                p_h, p_a = predict_match(elo_h, elo_a, neutral=neutral_sim)
                m_h, m_e, m_a = st.columns(3)
                m_h.metric(f"{sim_home}", f"{p_h*100:.1f}%")
                m_e.metric("Empate", f"{(1-p_h-p_a)*100:.1f}%")
                m_a.metric(f"{sim_away}", f"{p_a*100:.1f}%")
            else:
                st.warning("Selecione seleções diferentes.")

    st.markdown("---")

    # --- Distribuição de Elo e Variação (duas colunas) ---
    col_hist, col_var = st.columns([1, 1], gap="large")
    with col_hist:
        st.markdown("#### 📊 Distribuição de Elo")
        fig_hist = px.histogram(ranking_df, x="Elo", nbins=30,
                                color_discrete_sequence=[ACCENT],
                                marginal="box", title="Frequência de Ratings")
        fig_hist.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(t=40, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_var:
        st.markdown("#### 📉 Maiores Altas e Quedas (últimos 12 meses)")
        top_risers = ranking_df.nlargest(8, "Δ12m")[["Seleção", "Δ12m"]]
        top_fallers = ranking_df.nsmallest(8, "Δ12m")[["Seleção", "Δ12m"]]
        col_up, col_down = st.columns(2)
        with col_up:
            fig_up = px.bar(top_risers, x="Δ12m", y="Seleção", orientation="h",
                            title="Em ascensão", color="Δ12m", color_continuous_scale="Greens")
            fig_up.update_layout(template=PLOTLY_TEMPLATE, height=300, showlegend=False, margin=dict(t=30, b=10))
            st.plotly_chart(fig_up, use_container_width=True)
        with col_down:
            fig_down = px.bar(top_fallers, x="Δ12m", y="Seleção", orientation="h",
                              title="Em queda", color="Δ12m", color_continuous_scale="Reds")
            fig_down.update_layout(template=PLOTLY_TEMPLATE, height=300, showlegend=False, margin=dict(t=30, b=10))
            st.plotly_chart(fig_down, use_container_width=True)

    st.markdown("---")

    # --- Evolução histórica (gráfico de linha) ---
    st.subheader("📅 Evolução Histórica do Elo")
    with st.container(border=True):
        col_ts1, col_ts2 = st.columns([1, 1])
        with col_ts1:
            selected_teams_ts = st.multiselect(
                "Selecione seleções para comparar",
                ALL_TEAMS,
                default=["Brazil", "Germany", "Argentina"] if "Brazil" in ALL_TEAMS else ALL_TEAMS[:3],
                key="ts_teams"
            )
        with col_ts2:
            smooth_window = st.slider("Suavização (média móvel em jogos)", 1, 20, 5, help="Média móvel para suavizar a curva")
            start_date = HISTORY["date"].min().date()
            end_date = HISTORY["date"].max().date()
            date_range = st.slider("Período", min_value=start_date, max_value=end_date, value=(start_date, end_date))

    if selected_teams_ts:
        fig_ts = go.Figure()
        for team in selected_teams_ts:
            ts = get_elo_timeseries(HISTORY, team)
            if ts.empty:
                continue
            ts = ts[(ts["date"] >= pd.Timestamp(date_range[0])) & (ts["date"] <= pd.Timestamp(date_range[1]))]
            if smooth_window > 1:
                ts["elo_smooth"] = ts["elo"].rolling(smooth_window, min_periods=1).mean()
                y_vals = ts["elo_smooth"]
            else:
                y_vals = ts["elo"]
            fig_ts.add_trace(go.Scatter(x=ts["date"], y=y_vals, mode="lines", name=team, line=dict(width=2)))
        fig_ts.update_layout(template=PLOTLY_TEMPLATE, height=450,
                             title="Trajetória de Rating Elo (com suavização opcional)",
                             xaxis_title="Data", yaxis_title="Elo Rating",
                             hovermode="x unified")
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Selecione ao menos uma seleção para visualizar a evolução.")

    st.markdown("---")

    # --- Comparações históricas + Média por década (duas colunas) ---
    col_era1, col_era2 = st.columns([1, 1.2], gap="large")
    with col_era1:
        st.subheader("⏳ Comparações Históricas")
        team_era = st.selectbox("Escolha uma seleção", ALL_TEAMS, key="team_era")
        hist_team = HISTORY[HISTORY["team"] == team_era].sort_values("date")
        if not hist_team.empty:
            with st.container(border=True):
                max_elo = hist_team["elo"].max()
                min_elo = hist_team["elo"].min()
                current_elo = CURRENT_RATINGS[team_era]
                max_date = hist_team[hist_team["elo"] == max_elo]["date"].iloc[0].strftime("%Y-%m")
                min_date = hist_team[hist_team["elo"] == min_elo]["date"].iloc[0].strftime("%Y-%m")
                st.metric("Elo Atual", f"{current_elo:.0f}")
                st.metric("Pico Histórico", f"{max_elo:.0f} ({max_date})")
                st.metric("Mínimo Histórico", f"{min_elo:.0f} ({min_date})")
            compare_df = pd.DataFrame({
                "Métrica": ["Atual", "Pico", "Mínimo"],
                "Elo": [current_elo, max_elo, min_elo]
            })
            fig_comp = px.bar(compare_df, x="Métrica", y="Elo", color="Métrica",
                              color_discrete_sequence=[ACCENT, POSITIVE, NEGATIVE],
                              title=f"Desempenho: {team_era}")
            fig_comp.update_layout(template=PLOTLY_TEMPLATE, height=300, showlegend=False, margin=dict(t=30, b=10))
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Sem dados históricos suficientes.")

    with col_era2:
        st.subheader("📅 Média de Elo por Década (Top 10)")
        HISTORY_copy = HISTORY.copy()
        HISTORY_copy["decade"] = (HISTORY_copy["date"].dt.year // 10) * 10
        decades = sorted(HISTORY_copy["decade"].dropna().unique())
        selected_decade = st.selectbox("Selecione a década", decades, index=len(decades)-1)
        decade_data = HISTORY_copy[HISTORY_copy["decade"] == selected_decade]
        avg_by_team = decade_data.groupby("team")["elo"].mean().sort_values(ascending=False).head(10).reset_index()
        avg_by_team.columns = ["Seleção", "Elo Médio"]
        fig_decade = px.bar(avg_by_team, x="Elo Médio", y="Seleção", orientation="h",
                            color="Elo Médio", color_continuous_scale="Viridis",
                            title=f"Top 10 seleções na década de {int(selected_decade)}")
        fig_decade.update_layout(template=PLOTLY_TEMPLATE, height=500, margin=dict(t=40, b=10))
        st.plotly_chart(fig_decade, use_container_width=True)

    # --- Heatmap (expansível) ---
    with st.expander("🔥 Mapa de Calor da Evolução (seleções vs tempo)", expanded=False):
        st.caption("Exibe a variação de Elo ao longo dos anos para as principais seleções. Pode demorar um pouco.")
        n_heat = st.slider("Número de seleções no heatmap", 10, 50, 25, key="n_heat")
        top_teams_heat = ranking_df.head(n_heat)["Seleção"].tolist()
        history_heat = HISTORY[HISTORY["team"].isin(top_teams_heat)].copy()
        history_heat["year"] = history_heat["date"].dt.year
        heat_pivot = history_heat.groupby(["year", "team"])["elo"].mean().reset_index()
        heat_pivot = heat_pivot.pivot(index="team", columns="year", values="elo")
        heat_pivot = heat_pivot.ffill(axis=1).bfill(axis=1).fillna(INITIAL_ELO)
        heat_pivot = heat_pivot.reindex(ranking_df.head(n_heat)["Seleção"].tolist())
        fig_heat = go.Figure(data=go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns.astype(int),
            y=heat_pivot.index,
            colorscale="RdYlGn",
            zmid=INITIAL_ELO,
            colorbar=dict(title="Elo")
        ))
        fig_heat.update_layout(template=PLOTLY_TEMPLATE, height=500,
                               title="Evolução do Rating Elo por Seleção (média anual)",
                               xaxis_title="Ano", yaxis_title="Seleção")
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- Download do ranking ---
    csv_ranking = ranking_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar ranking atual (CSV)", data=csv_ranking, file_name="elo_ranking.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# TAB 2 — HEAD TO HEAD
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Confronto Direto")
    
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            team_a = st.selectbox("Seleção A", ALL_TEAMS, index=ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 0)
        with c2:
            default_b_idx = ALL_TEAMS.index("Argentina") if "Argentina" in ALL_TEAMS else 1
            team_b = st.selectbox("Seleção B", ALL_TEAMS, index=default_b_idx)

    h2h = RESULTS[
        ((RESULTS["home_team"] == team_a) & (RESULTS["away_team"] == team_b)) |
        ((RESULTS["home_team"] == team_b) & (RESULTS["away_team"] == team_a))
    ].sort_values("date", ascending=False)

    if h2h.empty:
        st.warning("Nenhum confronto direto registrado entre essas seleções.")
    else:
        wins_a = ((h2h["home_team"] == team_a) & (h2h["home_score"] > h2h["away_score"])).sum() + \
                 ((h2h["away_team"] == team_a) & (h2h["away_score"] > h2h["home_score"])).sum()
        wins_b = ((h2h["home_team"] == team_b) & (h2h["home_score"] > h2h["away_score"])).sum() + \
                 ((h2h["away_team"] == team_b) & (h2h["away_score"] > h2h["home_score"])).sum()
        draws = (h2h["home_score"] == h2h["away_score"]).sum()
        goals_a = h2h.apply(lambda r: r["home_score"] if r["home_team"] == team_a else r["away_score"], axis=1).sum()
        goals_b = h2h.apply(lambda r: r["home_score"] if r["home_team"] == team_b else r["away_score"], axis=1).sum()

        with st.container(border=True):
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric(f"Vitórias {team_a}", int(wins_a))
            m2.metric("Empates", int(draws))
            m3.metric(f"Vitórias {team_b}", int(wins_b))
            m4.metric(f"Gols {team_a}", int(goals_a))
            m5.metric(f"Gols {team_b}", int(goals_b))

        elo_a_val = CURRENT_RATINGS.get(team_a, INITIAL_ELO)
        elo_b_val = CURRENT_RATINGS.get(team_b, INITIAL_ELO)
        tier_a, color_a = elo_tier(elo_a_val)
        tier_b, color_b = elo_tier(elo_b_val)

        with st.container(border=True):
            cforma, cformb = st.columns(2)
            with cforma:
                st.markdown(
                    f"**{team_a}** — Elo `{elo_a_val:.1f}` "
                    f'<span class="tier-badge" style="border-color:{color_a};color:{color_a};">{tier_a}</span><br>'
                    f"Forma recente: {form_pills_html(RESULTS, team_a)}",
                    unsafe_allow_html=True
                )
            with cformb:
                st.markdown(
                    f"**{team_b}** — Elo `{elo_b_val:.1f}` "
                    f'<span class="tier-badge" style="border-color:{color_b};color:{color_b};">{tier_b}</span><br>'
                    f"Forma recente: {form_pills_html(RESULTS, team_b)}",
                    unsafe_allow_html=True
                )

        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            pie = go.Figure(data=[go.Pie(
                labels=[f"{team_a}", "Empate", f"{team_b}"],
                values=[wins_a, draws, wins_b],
                hole=0.5,
                marker=dict(colors=[ACCENT, "#999999", ACCENT2]),
            )])
            pie.update_layout(template=PLOTLY_TEMPLATE, title="Distribuição de Resultados", height=400)
            st.plotly_chart(pie, use_container_width=True)

        with col2:
            h2h_sorted = h2h.sort_values("date")
            h2h_sorted["goal_diff"] = h2h_sorted.apply(
                lambda r: (r["home_score"] - r["away_score"]) if r["home_team"] == team_a
                else (r["away_score"] - r["home_score"]), axis=1
            )
            bar = go.Figure()
            colors = [ACCENT if v > 0 else (ACCENT2 if v < 0 else "#999999") for v in h2h_sorted["goal_diff"]]
            bar.add_trace(go.Bar(x=h2h_sorted["date"], y=h2h_sorted["goal_diff"], marker_color=colors))
            bar.update_layout(template=PLOTLY_TEMPLATE, title=f"Saldo de Gols por Confronto (perspectiva {team_a})", height=400,
                               yaxis_title="Saldo de gols")
            st.plotly_chart(bar, use_container_width=True)

        st.markdown("---")
        
        col_tables1, col_tables2 = st.columns(2, gap="large")
        
        with col_tables1:
            st.markdown("**Histórico de Confrontos**")
            st.dataframe(
                h2h[["date", "home_team", "away_team", "home_score", "away_score", "tournament"]],
                use_container_width=True, hide_index=True, height=350
            )

        with col_tables2:
            gs = DATA["goalscorers"]
            gs_h2h = gs[
                ((gs["home_team"] == team_a) & (gs["away_team"] == team_b)) |
                ((gs["home_team"] == team_b) & (gs["away_team"] == team_a))
            ]
            gs_h2h = gs_h2h[gs_h2h["own_goal"] != True]
            if not gs_h2h.empty:
                top_scorers = gs_h2h.groupby(["scorer", "team"]).size().reset_index(name="Gols").sort_values("Gols", ascending=False).head(10)
                st.markdown("**Maiores Artilheiros no Confronto**")
                st.dataframe(top_scorers, use_container_width=True, hide_index=True, height=350)
            else:
                st.info("Sem dados de artilharia detalhados para o confronto.")

        so = DATA["shootouts"]
        so_h2h = so[
            ((so["home_team"] == team_a) & (so["away_team"] == team_b)) |
            ((so["home_team"] == team_b) & (so["away_team"] == team_a))
        ]
        if not so_h2h.empty:
            with st.container(border=True):
                st.markdown("**Disputas de Pênaltis**")
                st.dataframe(so_h2h[["date", "home_team", "away_team", "winner"]], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 3 — POISSON PREDICTION
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Modelo de Predição (Poisson Bivariado)")
    st.markdown(
        '<div class="caption-box">Estimativa de força ofensiva/defensiva calculada a partir dos últimos 10 anos de jogos. '
        'λ (lambda) representa o número esperado de gols de cada seleção.</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            home_pred = st.selectbox("Seleção da casa", ALL_TEAMS, key="home_pred",
                                      index=ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 0)
        with c2:
            away_pred = st.selectbox("Seleção visitante", ALL_TEAMS, key="away_pred",
                                      index=ALL_TEAMS.index("Argentina") if "Argentina" in ALL_TEAMS else 1)
        with c3:
            neutral_pred = st.checkbox("Campo neutro", value=True)

    if home_pred == away_pred:
        st.warning("Selecione seleções diferentes.")
    else:
        result = poisson_match_probs(home_pred, away_pred, STRENGTH, DATA["avg_home"], DATA["avg_away"])
        elo_h = CURRENT_RATINGS.get(home_pred, INITIAL_ELO)
        elo_a = CURRENT_RATINGS.get(away_pred, INITIAL_ELO)
        p_elo_home, p_elo_away = predict_match(elo_h, elo_a, neutral=neutral_pred)

        with st.container(border=True):
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Vitória {home_pred}", f"{result['p_home_win']*100:.1f}%")
            m2.metric("Empate", f"{result['p_draw']*100:.1f}%")
            m3.metric(f"Vitória {away_pred}", f"{result['p_away_win']*100:.1f}%")

            gauge = go.Figure()
            gauge.add_trace(go.Bar(
                x=[result["p_home_win"]*100], y=["Probabilidade"], orientation="h",
                name=home_pred, marker_color=ACCENT, text=f"{result['p_home_win']*100:.0f}%", textposition="inside",
            ))
            gauge.add_trace(go.Bar(
                x=[result["p_draw"]*100], y=["Probabilidade"], orientation="h",
                name="Empate", marker_color="#888", text=f"{result['p_draw']*100:.0f}%", textposition="inside",
            ))
            gauge.add_trace(go.Bar(
                x=[result["p_away_win"]*100], y=["Probabilidade"], orientation="h",
                name=away_pred, marker_color=ACCENT2, text=f"{result['p_away_win']*100:.0f}%", textposition="inside",
            ))
            gauge.update_layout(
                template=PLOTLY_TEMPLATE, barmode="stack", height=120,
                showlegend=True, xaxis=dict(range=[0, 100], showticklabels=False),
                yaxis=dict(showticklabels=False), margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.05),
            )
            st.plotly_chart(gauge, use_container_width=True)

            st.markdown("---")
            m4, m5, m6 = st.columns(3)
            m4.metric(f"λ gols {home_pred}", f"{result['lambda_home']:.2f}")
            m5.metric(f"λ gols {away_pred}", f"{result['lambda_away']:.2f}")
            m6.metric("Elo (referência)", f"{elo_h:.0f} vs {elo_a:.0f}")

            st.caption(f"Probabilidade implícita por Elo: {home_pred} {p_elo_home*100:.1f}% — {away_pred} {p_elo_away*100:.1f}% (modelo simplificado)")

        col_heat, col_scores = st.columns([2, 1], gap="large")
        
        with col_heat:
            max_g = 6
            sm = result["score_matrix"][:max_g+1, :max_g+1]
            heat = go.Figure(data=go.Heatmap(
                z=sm, x=[str(i) for i in range(max_g+1)], y=[str(i) for i in range(max_g+1)],
                colorscale="Blues",
                text=np.round(sm*100, 1), texttemplate="%{text}%",
                hoverongaps=False,
            ))
            heat.update_layout(
                template=PLOTLY_TEMPLATE,
                title=f"Matriz de Probabilidade de Placar",
                xaxis_title=f"Gols {away_pred}", yaxis_title=f"Gols {home_pred}",
                height=450,
                margin=dict(t=50, b=10)
            )
            st.plotly_chart(heat, use_container_width=True)

        with col_scores:
            flat = []
            for i in range(max_g+1):
                for j in range(max_g+1):
                    flat.append({"Placar": f"{i} x {j}", "Prob": sm[i, j]})
            top_scores = pd.DataFrame(flat).sort_values("Prob", ascending=False).head(10)
            top_scores["Prob"] = (top_scores["Prob"] * 100).round(1).astype(str) + "%"
            st.markdown("#### Placares mais prováveis")
            st.dataframe(top_scores, use_container_width=True, hide_index=True, height=380)


# ---------------------------------------------------------------------------
# TAB 4 — TOURNAMENT ANALYTICS
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Análise por Competição")

    torneios = sorted(RESULTS["tournament"].dropna().unique())
    selected_tournament = st.selectbox("Escolha uma competição", torneios)

    df_tourn = RESULTS[RESULTS["tournament"] == selected_tournament].copy()
    if df_tourn.empty:
        st.warning("Nenhuma partida encontrada.")
    else:
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            total_jogos = len(df_tourn)
            total_gols = df_tourn["home_score"].sum() + df_tourn["away_score"].sum()
            media_gols = total_gols / total_jogos
            times_distintos = pd.concat([df_tourn["home_team"], df_tourn["away_team"]]).nunique()

            col1.metric("Total de partidas", total_jogos)
            col2.metric("Total de gols", total_gols)
            col3.metric("Média de gols/jogo", f"{media_gols:.2f}")
            col4.metric("Países participantes", times_distintos)

        col_t1, col_t2 = st.columns([1, 2], gap="large")
        
        with col_t1:
            gs = DATA["goalscorers"]
            gs_tourn = gs.merge(
                df_tourn[["date", "home_team", "away_team"]],
                on=["date", "home_team", "away_team"],
                how="inner"
            )
            gs_tourn = gs_tourn[gs_tourn["own_goal"] != True]
            if not gs_tourn.empty:
                top_scorers = (
                    gs_tourn.groupby("scorer")
                    .size()
                    .reset_index(name="gols")
                    .sort_values("gols", ascending=False)
                    .head(10)
                )
                st.markdown("#### Artilheiros Históricos")
                st.dataframe(top_scorers, use_container_width=True, hide_index=True, height=400)
            else:
                st.info("Sem dados de artilharia detalhados para o torneio selecionado.")

        with col_t2:
            df_tourn["year"] = df_tourn["date"].dt.year
            gols_por_ano = df_tourn.groupby("year").apply(
                lambda x: x["home_score"].sum() + x["away_score"].sum()
            ).reset_index(name="gols")
            fig = px.line(gols_por_ano, x="year", y="gols", title="Evolução de gols por ano")
            fig.update_layout(template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 5 — PLAYER MARKET INTELLIGENCE
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Inteligência de Mercado — Jogadores & Transferências")
    st.markdown(
        '<div class="caption-box">Fonte: Transfermarkt (clubes, jogadores, valores de mercado e transferências). '
        'Cruzamento com o Elo Engine permite conectar o valor de mercado dos clubes às seleções nacionais.</div>',
        unsafe_allow_html=True
    )

    if CLUB_DATA is None:
        st.error("Dados de clubes não disponíveis. Verifique o diretório 'data_clubs/' com os CSVs necessários.")
    else:
        PLAYERS = CLUB_DATA["players"]
        CLUBS = CLUB_DATA["clubs"]
        TRANSFERS = CLUB_DATA["transfers"]
        VALUATIONS = CLUB_DATA["valuations"]
        COMPETITIONS = CLUB_DATA["competitions"]

        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Jogadores na base", f"{len(PLAYERS):,}".replace(",", "."))
            m2.metric("Clubes", f"{len(CLUBS):,}".replace(",", "."))
            m3.metric("Transferências registradas", f"{len(TRANSFERS):,}".replace(",", "."))
            total_fees = TRANSFERS["transfer_fee"].sum()
            m4.metric("Volume total negociado", f"€ {total_fees/1e9:.1f} bi")

        st.markdown("---")
        # 1. Fluxo de transferências entre ligas
        st.markdown("#### 🌍 Fluxo de Transferências entre Ligas")
        seasons = sorted(TRANSFERS["transfer_season"].dropna().unique(), reverse=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            season_sel = st.selectbox("Temporada", ["Todas"] + list(seasons))
            min_fee_m = st.slider("Valor mínimo por transferência (€ milhões)", 0.5, 20.0, 2.0, step=0.5)

        flow = transfer_flow_by_league(
            TRANSFERS, CLUBS,
            season=None if season_sel == "Todas" else season_sel,
            top_n=15, min_fee=min_fee_m * 1_000_000
        )
        with col2:
            if flow.empty:
                st.info("Sem fluxos relevantes para os filtros selecionados.")
            else:
                labels = list(pd.unique(flow[["from_league", "to_league"]].values.ravel()))
                label_to_idx = {l: i for i, l in enumerate(labels)}
                sources = flow["from_league"].map(label_to_idx).tolist()
                targets = flow["to_league"].map(label_to_idx).tolist()
                values = flow["transfer_fee"].tolist()
                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(label=labels, pad=15, thickness=18, color=ACCENT),
                    link=dict(source=sources, target=targets, value=[v / 1e6 for v in values], color="rgba(212,175,55,0.35)")
                )])
                fig_sankey.update_layout(template=PLOTLY_TEMPLATE, height=450,
                                         title=f"Fluxo de investimento – {season_sel}", margin=dict(t=40, b=10))
                st.plotly_chart(fig_sankey, use_container_width=True)

        st.markdown("---")
        # 2. Curva de valorização por idade
        st.markdown("#### 📈 Curva de Valorização de Mercado por Idade")
        col3, col4 = st.columns([1, 3])
        with col3:
            position_sel = st.selectbox("Posição", ["Todas", "Attack", "Midfield", "Defender", "Goalkeeper"])
        curve = valuation_age_curve(VALUATIONS, PLAYERS, position_filter=position_sel)
        with col4:
            if curve.empty:
                st.info("Dados insuficientes para esta posição.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=curve["age_bucket"], y=curve["mean"]/1e6, mode="lines+markers",
                                         name="Valor médio (€ mi)", line=dict(color=ACCENT, width=2)))
                fig.add_trace(go.Scatter(x=curve["age_bucket"], y=curve["median"]/1e6, mode="lines+markers",
                                         name="Valor mediano (€ mi)", line=dict(color=ACCENT2, width=2, dash="dot")))
                peak_age = curve.loc[curve["mean"].idxmax(), "age_bucket"]
                fig.add_vline(x=peak_age, line_dash="dash", line_color="#999999",
                              annotation_text=f"Pico ~ {int(peak_age)} anos")
                fig.update_layout(template=PLOTLY_TEMPLATE, height=400,
                                  xaxis_title="Idade", yaxis_title="Valor de mercado (€ milhões)", margin=dict(t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        # 3. Maiores transferências + nacionalidades
        col5, col6 = st.columns(2, gap="large")
        with col5:
            st.markdown("#### 💸 Maiores Transferências da História")
            tt = top_transfers(TRANSFERS, n=15, min_fee=1_000_000)
            tt_display = tt.copy()
            tt_display["transfer_fee"] = (tt_display["transfer_fee"] / 1e6).round(1).astype(str) + " M€"
            tt_display.columns = ["Data", "Jogador", "De", "Para", "Valor", "Temporada"]
            st.dataframe(tt_display, use_container_width=True, hide_index=True, height=400)

        with col6:
            st.markdown("#### 🌎 Nacionalidade dos Jogadores por Liga")
            comp_options = ["Todas"] + sorted(PLAYERS["current_club_domestic_competition_id"].dropna().unique().tolist())
            comp_labels = {c: COMPETITION_LABELS.get(c, c) for c in comp_options}
            comp_sel = st.selectbox("Liga", comp_options, format_func=lambda x: comp_labels.get(x, x))
            nat = nationality_distribution(PLAYERS, comp_sel).head(12)
            fig = px.bar(nat, x="Jogadores", y="País", orientation="h")
            fig.update_traces(marker_color=ACCENT)
            fig.update_layout(template=PLOTLY_TEMPLATE, height=380, yaxis=dict(autorange="reversed"), margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        col_c1, col_c2 = st.columns(2, gap="large")
        # 4. Explorador de clubes
        with col_c1:
            st.markdown("#### 🏟️ Explorador de Clubes")
            league_options = ["Todas"] + sorted(CLUBS["domestic_competition_id"].dropna().unique().tolist())
            league_labels = {l: COMPETITION_LABELS.get(l, l) for l in league_options}
            league_sel = st.selectbox("Filtrar por liga", league_options, format_func=lambda x: league_labels.get(x, x), key="club_league")
            cs = club_summary(CLUBS, PLAYERS, league_sel).sort_values("squad_value_eur", ascending=False).head(20)
            cs_display = cs.copy()
            cs_display["squad_value_eur"] = (cs_display["squad_value_eur"] / 1e6).round(1).astype(str) + " M€"
            cs_display.columns = ["Clube", "Elenco", "Idade Média", "% Estrangeiros", "Jogadores Sel. Nacional",
                                  "Valor do Elenco", "Estádio", "Capacidade"]
            st.dataframe(cs_display, use_container_width=True, hide_index=True, height=400)

        # 5. Conexão Seleção ↔ Clube
        with col_c2:
            st.markdown("#### 🔗 Conexão Seleção ↔ Clube")
            bridge_team = st.selectbox("Seleção", sorted(PLAYERS["country_of_citizenship"].dropna().unique()),
                                       index=0, key="bridge_team")
            bridge = PLAYERS[PLAYERS["country_of_citizenship"] == bridge_team].copy()
            bridge = bridge[bridge["international_caps"].notna() & (bridge["international_caps"] > 0)]
            bridge = bridge.sort_values("international_caps", ascending=False).head(15)
            if bridge.empty:
                st.info("Sem dados internacionais suficientes para esta seleção na base.")
            else:
                bridge_display = bridge[["name", "current_club_name", "position", "international_caps",
                                         "international_goals", "market_value_in_eur"]].copy()
                bridge_display["market_value_in_eur"] = (bridge_display["market_value_in_eur"] / 1e6).round(1).astype(str) + " M€"
                bridge_display.columns = ["Jogador", "Clube Atual", "Posição", "Jogos Pela Seleção", "Gols", "Valor de Mercado"]
                st.dataframe(bridge_display, use_container_width=True, hide_index=True, height=400)


# ---------------------------------------------------------------------------
# TAB 6 — METODOLOGIA
# ---------------------------------------------------------------------------
with tab6:
    st.subheader("Metodologia e Fontes")
    st.markdown("""
    ### 🔢 **Elo Ratings**
    - Baseado no sistema **World Football Elo Ratings** com fatores dinâmicos:
      - Importância do jogo (pesos por torneio)
      - Vantagem de jogar em casa (+50 pontos)
      - Goleada (multiplicador baseado na diferença de gols)
    - Rating inicial: **1500** para todas as seleções.

    ### 📈 **Modelo Poisson Bivariado**
    - Estima força ofensiva e defensiva de cada seleção a partir dos últimos 10 anos.
    - Gera probabilidades de vitória/empate/derrota e matriz de placares esperados.

    ### 💰 **Inteligência de Mercado (Transfermarkt)**
    - Dados de **jogadores, clubes, transferências e valuation** fornecidos pelo Kaggle.
    - Curva de valorização por idade, maiores transferências, fluxo financeiro entre ligas, distribuição de nacionalidades e resumo financeiro dos clubes.

    ### 🗺️ **Estádios**
    - Coordenadas geográficas via Wikidata (SPARQL) + enriquecimento com capacidade.
    - Mapa interativo baseado em OpenStreetMap.

    ### 📂 **Fontes**
    - Partidas internacionais: [Football Results (1872-2024)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
    - Dados de mercado: [Transfermarkt Dataset](https://www.kaggle.com/datasets/davidcariboo/player-scores)
    - Estádios: Wikidata SPARQL endpoint + Kaggle Football Stadiums
    - Processamento e visualização: **Python, Pandas, Plotly, Streamlit, Folium**.

    ---
    **Desenvolvido por Eduardo Moraes** [GitHub](https://github.com/edu-moraess/global-football-elo)
    """)


# ---------------------------------------------------------------------------
# TAB 7 — ESTÁDIOS (MAPA REAL COM FOLIUM + OPENSTREETMAP)
# ---------------------------------------------------------------------------
with tab7:
    st.subheader("🗺️ Mapa Mundial de Estádios")
    st.markdown("Use o scroll para zoom e arraste para navegar. Mapa baseado em OpenStreetMap.")

    @st.cache_data
    def load_stadiums_data():
        path = Path("data/top_1000_stadiums_world.csv")
        if not path.exists():
            st.warning("Arquivo 'top_1000_stadiums_world.csv' não encontrado em data/")
            return pd.DataFrame()

        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin1')

        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])

        df["team"] = df["team"].fillna("Desconhecido")
        df["league"] = df["league"].fillna("Desconhecida")
        df["country"] = df["country"].fillna("Desconhecido")

        df = df.rename(columns={
            "stadium": "Stadium",
            "team": "Team",
            "league": "League",
            "country": "Country",
            "latitude": "Latitude",
            "longitude": "Longitude"
        })

        # Capacidade (se disponível)
        cap_path = Path("data/Football Stadiums.csv")
        if cap_path.exists():
            try:
                cap_df = pd.read_csv(cap_path, encoding='utf-8')
            except UnicodeDecodeError:
                cap_df = pd.read_csv(cap_path, encoding='latin1')
            cap_df["Stadium_clean"] = cap_df["Stadium"].str.strip().str.lower()
            df["Stadium_clean"] = df["Stadium"].str.strip().str.lower()
            df = df.merge(cap_df[["Stadium_clean", "Capacity"]], on="Stadium_clean", how="left")
            df.drop(columns=["Stadium_clean"], inplace=True)
            if "Capacity" in df.columns:
                df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce").fillna(0).astype(int)
                df["Capacity_display"] = df["Capacity"].apply(lambda x: f"{x:,}" if x > 0 else "N/A")
            else:
                df["Capacity"] = 0
                df["Capacity_display"] = "N/A"
        else:
            df["Capacity"] = 0
            df["Capacity_display"] = "N/A"

        return df

    df_stadiums = load_stadiums_data()

    if not df_stadiums.empty:
        with st.container(border=True):
            st.markdown("#### 🔍 Filtros do mapa")
            col1, col2 = st.columns([1, 2])
            with col1:
                pais_mapa = st.selectbox(
                    "País em destaque",
                    ["Todos"] + sorted(df_stadiums["Country"].unique())
                )
            with col2:
                busca = st.text_input("Buscar estádio por nome", "")

        df_plot = df_stadiums.copy()
        if pais_mapa != "Todos":
            df_plot = df_plot[df_plot["Country"] == pais_mapa]
        if busca:
            df_plot = df_plot[df_plot["Stadium"].str.contains(busca, case=False)]

        if not df_plot.empty:
            center_lat = df_plot["Latitude"].mean()
            center_lon = df_plot["Longitude"].mean()

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=2,
                tiles='OpenStreetMap',
                control_scale=True,
                attr='OpenStreetMap contributors'
            )

            folium.TileLayer(
                tiles='CartoDB positron',
                name='Claro Minimalista',
                attr='CartoDB'
            ).add_to(m)

            folium.TileLayer(
                tiles='OpenStreetMap',
                name='Ruas',
                attr='OpenStreetMap contributors'
            ).add_to(m)

            marker_cluster = MarkerCluster(name="Estádios").add_to(m)

            for _, row in df_plot.iterrows():
                popup_html = f"""
                <b>{row['Stadium']}</b><br>
                Time: {row['Team']}<br>
                Liga: {row['League']}<br>
                País: {row['Country']}<br>
                Capacidade: {row['Capacity_display']}
                """
                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color='blue', icon='futbol-o', prefix='fa')
                ).add_to(marker_cluster)

            folium.LayerControl().add_to(m)
            Fullscreen().add_to(m)

            st_folium(m, width=1400, height=700, returned_objects=[])

            st.caption(f"Mostrando {len(df_plot)} estádios. Clique em um marcador para detalhes.")
        else:
            st.info("Nenhum estádio encontrado com os filtros atuais.")

        st.markdown("---")
        st.subheader("📋 Explorador de Estádios")

        col1, col2 = st.columns(2)
        with col1:
            paises = sorted(df_stadiums["Country"].unique())
            pais_sel = st.selectbox("Filtrar por país (tabela)", ["Todos"] + paises)
        with col2:
            if "Team" in df_stadiums.columns:
                times = sorted(df_stadiums["Team"].unique())
                time_sel = st.selectbox("Filtrar por time mandante", ["Todos"] + times)
            else:
                time_sel = "Todos"

        df_filtrado = df_stadiums.copy()
        if pais_sel != "Todos":
            df_filtrado = df_filtrado[df_filtrado["Country"] == pais_sel]
        if time_sel != "Todos" and "Team" in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado["Team"] == time_sel]

        colunas_exibir = ["Stadium", "Team", "League", "Country", "Capacity_display"]
        colunas_disponiveis = [c for c in colunas_exibir if c in df_filtrado.columns]
        df_display = df_filtrado[colunas_disponiveis].rename(columns={"Capacity_display": "Capacidade"})

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        st.caption(f"Mostrando {len(df_display)} estádios de um total de {len(df_stadiums)} com coordenadas.")
    else:
        st.info("Nenhum dado de estádio com coordenadas disponível. Adicione o arquivo 'top_1000_stadiums_world.csv' na pasta data/")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    f'<div style="text-align:center; font-size:0.75rem; opacity:0.6; margin-top:2rem; '
    f'font-family:\'IBM Plex Mono\',monospace;">Global Football Intelligence — Quant Analytics</div>',
    unsafe_allow_html=True
)