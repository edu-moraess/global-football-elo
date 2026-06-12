"""
Edumetria | Global Football Intelligence
Painel quantitativo: Elo Ratings, Head-to-Head, Predição Poisson
Dataset: International football results (1872-2024)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from elo_engine import (
    load_former_names, build_name_map, normalize_team_names,
    compute_elo_history, get_elo_timeseries, predict_match,
    team_attack_defense_strength, poisson_match_probs, INITIAL_ELO
)

# ---------------------------------------------------------------------------
# CONFIG & THEME
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Edumetria | Global Football Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_BG = "#0d1117"
PANEL_BG = "#161b22"
ACCENT = "#d4af37"   # gold
ACCENT2 = "#4cc9f0"  # cyan accent
TEXT_MAIN = "#e6edf3"
TEXT_MUTED = "#8b949e"
GRID_COLOR = "#21262d"

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=PRIMARY_BG,
        plot_bgcolor=PRIMARY_BG,
        font=dict(family="IBM Plex Mono, monospace", color=TEXT_MAIN, size=12),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        colorway=[ACCENT, ACCENT2, "#f72585", "#4ade80", "#fb923c", "#a78bfa"],
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)

st.markdown(f"""
<style>
.stApp {{ background-color: {PRIMARY_BG}; }}
section[data-testid="stSidebar"] {{ background-color: {PANEL_BG}; border-right: 1px solid {GRID_COLOR}; }}
h1, h2, h3 {{ font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.5px; }}
h1 {{ color: {ACCENT} !important; border-bottom: 1px solid {GRID_COLOR}; padding-bottom: 12px; }}
h2, h3 {{ color: {TEXT_MAIN} !important; }}
[data-testid="stMetricValue"] {{ color: {ACCENT}; font-family: 'IBM Plex Mono', monospace; }}
[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED}; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'IBM Plex Mono', monospace; color: {TEXT_MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {ACCENT} !important; }}
.block-container {{ padding-top: 1.5rem; }}
[data-testid="stDataFrame"] {{ border: 1px solid {GRID_COLOR}; }}
.caption-box {{
    background-color: {PANEL_BG}; border-left: 3px solid {ACCENT};
    padding: 0.6rem 1rem; font-size: 0.85rem; color: {TEXT_MUTED};
    font-family: 'IBM Plex Mono', monospace; margin-bottom: 1rem;
}}
</style>
""", unsafe_allow_html=True)


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
    f'<div class="caption-box">EDUMETRIA RESEARCH &nbsp;|&nbsp; '
    f'Elo Rating Engine &middot; Head-to-Head Analytics &middot; Modelo Poisson de Predição &nbsp;|&nbsp; '
    f'Base histórica: {RESULTS["date"].min().date()} — {RESULTS["date"].max().date()} '
    f'&nbsp;|&nbsp; {len(RESULTS):,} partidas</div>'.replace(",", "."),
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Elo Ranking & Evolução", "🆚 Head-to-Head", "🎯 Predição Poisson", "ℹ️ Metodologia"
])


# ---------------------------------------------------------------------------
# TAB 1 — ELO RANKING
# ---------------------------------------------------------------------------
with tab1:
    col_a, col_b = st.columns([1, 2])

    ranking_df = pd.DataFrame(
        [{"Seleção": k, "Elo": round(v, 1)} for k, v in CURRENT_RATINGS.items()]
    ).sort_values("Elo", ascending=False).reset_index(drop=True)
    ranking_df.index += 1
    ranking_df.index.name = "Rank"

    with col_a:
        st.subheader("Ranking Atual")
        top_n = st.slider("Top N seleções", 5, 50, 20)
        st.dataframe(
            ranking_df.head(top_n),
            use_container_width=True,
            height=600,
        )

    with col_b:
        st.subheader("Evolução Histórica do Elo")
        default_teams = [t for t in ["Brazil", "Germany", "Argentina", "France", "England"] if t in ALL_TEAMS]
        selected_teams = st.multiselect(
            "Selecione seleções para comparar",
            ALL_TEAMS,
            default=default_teams or ALL_TEAMS[:5],
        )

        if selected_teams:
            fig = go.Figure()
            for team in selected_teams:
                ts = get_elo_timeseries(HISTORY, team)
                if ts.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=ts["date"], y=ts["elo"], mode="lines", name=team, line=dict(width=1.8)
                ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                height=560,
                title="Trajetória de Rating Elo (1872 — presente)",
                xaxis_title="Data", yaxis_title="Elo Rating",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selecione ao menos uma seleção.")

        # Top movers (últimos 2 anos)
        st.subheader("Maiores Variações de Rating (últimos 24 meses)")
        cutoff = HISTORY["date"].max() - pd.DateOffset(months=24)
        recent = HISTORY[HISTORY["date"] >= cutoff]
        movers = []
        for team in ALL_TEAMS:
            ts = recent[recent["team"] == team].sort_values("date")
            if len(ts) >= 2:
                delta = ts["elo"].iloc[-1] - ts["elo"].iloc[0]
                movers.append({"Seleção": team, "Δ Elo (24m)": round(delta, 1), "Elo Atual": round(CURRENT_RATINGS[team], 1)})
        movers_df = pd.DataFrame(movers)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🔺 Em ascensão**")
            st.dataframe(movers_df.sort_values("Δ Elo (24m)", ascending=False).head(10), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**🔻 Em queda**")
            st.dataframe(movers_df.sort_values("Δ Elo (24m)").head(10), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 2 — HEAD TO HEAD
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Confronto Direto")
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

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(f"Vitórias {team_a}", int(wins_a))
        m2.metric("Empates", int(draws))
        m3.metric(f"Vitórias {team_b}", int(wins_b))
        m4.metric(f"Gols {team_a}", int(goals_a))
        m5.metric(f"Gols {team_b}", int(goals_b))

        st.markdown(f"**Elo atual** — {team_a}: `{CURRENT_RATINGS.get(team_a, INITIAL_ELO):.1f}` &nbsp;|&nbsp; {team_b}: `{CURRENT_RATINGS.get(team_b, INITIAL_ELO):.1f}`")

        # Pizza de resultados + linha temporal de gols
        col1, col2 = st.columns([1, 2])
        with col1:
            pie = go.Figure(data=[go.Pie(
                labels=[f"{team_a}", "Empate", f"{team_b}"],
                values=[wins_a, draws, wins_b],
                hole=0.5,
                marker=dict(colors=[ACCENT, GRID_COLOR, ACCENT2]),
            )])
            pie.update_layout(template=PLOTLY_TEMPLATE, title="Distribuição de Resultados", height=350)
            st.plotly_chart(pie, use_container_width=True)

        with col2:
            h2h_sorted = h2h.sort_values("date")
            h2h_sorted["goal_diff"] = h2h_sorted.apply(
                lambda r: (r["home_score"] - r["away_score"]) if r["home_team"] == team_a
                else (r["away_score"] - r["home_score"]), axis=1
            )
            bar = go.Figure()
            colors = [ACCENT if v > 0 else (ACCENT2 if v < 0 else GRID_COLOR) for v in h2h_sorted["goal_diff"]]
            bar.add_trace(go.Bar(x=h2h_sorted["date"], y=h2h_sorted["goal_diff"], marker_color=colors))
            bar.update_layout(template=PLOTLY_TEMPLATE, title=f"Saldo de Gols por Confronto (perspectiva {team_a})", height=350,
                               yaxis_title="Saldo de gols")
            st.plotly_chart(bar, use_container_width=True)

        # Shootouts
        so = DATA["shootouts"]
        so_h2h = so[
            ((so["home_team"] == team_a) & (so["away_team"] == team_b)) |
            ((so["home_team"] == team_b) & (so["away_team"] == team_a))
        ]
        if not so_h2h.empty:
            st.markdown("**Disputas de pênaltis**")
            st.dataframe(so_h2h[["date", "home_team", "away_team", "winner"]], use_container_width=True, hide_index=True)

        st.markdown("**Histórico de Confrontos**")
        st.dataframe(
            h2h[["date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country"]],
            use_container_width=True, hide_index=True, height=300
        )

        # Top scorers in this matchup
        gs = DATA["goalscorers"]
        gs_h2h = gs[
            ((gs["home_team"] == team_a) & (gs["away_team"] == team_b)) |
            ((gs["home_team"] == team_b) & (gs["away_team"] == team_a))
        ]
        gs_h2h = gs_h2h[gs_h2h["own_goal"] != True]
        if not gs_h2h.empty:
            top_scorers = gs_h2h.groupby(["scorer", "team"]).size().reset_index(name="Gols").sort_values("Gols", ascending=False).head(10)
            st.markdown("**Maiores artilheiros no confronto**")
            st.dataframe(top_scorers, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 3 — POISSON PREDICTION
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Modelo de Predição (Poisson Bivariado)")
    st.markdown(
        '<div class="caption-box">Estimativa de força ofensiva/defensiva calculada a partir dos últimos 10 anos de jogos. '
        'λ (lambda) representa o número esperado de gols de cada seleção, usado para gerar a matriz de probabilidade de placares.</div>',
        unsafe_allow_html=True
    )

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

        # Elo-based probabilities for comparison
        elo_h = CURRENT_RATINGS.get(home_pred, INITIAL_ELO)
        elo_a = CURRENT_RATINGS.get(away_pred, INITIAL_ELO)
        p_elo_home, p_elo_away = predict_match(elo_h, elo_a, neutral=neutral_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Vitória {home_pred}", f"{result['p_home_win']*100:.1f}%")
        m2.metric("Empate", f"{result['p_draw']*100:.1f}%")
        m3.metric(f"Vitória {away_pred}", f"{result['p_away_win']*100:.1f}%")

        m4, m5, m6 = st.columns(3)
        m4.metric(f"λ gols {home_pred}", f"{result['lambda_home']:.2f}")
        m5.metric(f"λ gols {away_pred}", f"{result['lambda_away']:.2f}")
        m6.metric("Elo (referência)", f"{elo_h:.0f} vs {elo_a:.0f}")

        st.caption(f"Probabilidade implícita por Elo: {home_pred} {p_elo_home*100:.1f}% — {away_pred} {p_elo_away*100:.1f}% (modelo simplificado, sem considerar empate)")

        # Score matrix heatmap
        max_g = 6
        sm = result["score_matrix"][:max_g+1, :max_g+1]
        heat = go.Figure(data=go.Heatmap(
            z=sm, x=[str(i) for i in range(max_g+1)], y=[str(i) for i in range(max_g+1)],
            colorscale=[[0, PRIMARY_BG], [1, ACCENT]],
            text=np.round(sm*100, 1), texttemplate="%{text}%",
            hoverongaps=False,
        ))
        heat.update_layout(
            template=PLOTLY_TEMPLATE,
            title=f"Matriz de Probabilidade de Placar — {home_pred} (linhas) x {away_pred} (colunas)",
            xaxis_title=f"Gols {away_pred}", yaxis_title=f"Gols {home_pred}",
            height=500,
        )
        st.plotly_chart(heat, use_container_width=True)

        # Most likely scorelines
        flat = []
        for i in range(max_g+1):
            for j in range(max_g+1):
                flat.append({"Placar": f"{i} x {j}", "Prob": sm[i, j]})
        top_scores = pd.DataFrame(flat).sort_values("Prob", ascending=False).head(5)
        top_scores["Prob"] = (top_scores["Prob"] * 100).round(1).astype(str) + "%"
        st.markdown("**Placares mais prováveis**")
        st.dataframe(top_scores, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 4 — METHODOLOGY
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Metodologia")
    st.markdown("""
**Fonte de dados**: International football results, 1872 a 2024 (results, goalscorers, shootouts, former_names).

**Elo Rating Engine**
- Rating inicial de 1500 para todas as seleções.
- Vantagem de campo de +50 pontos quando o jogo não é em campo neutro.
- Multiplicador de K por importância do torneio (Copa do Mundo: 60, Eliminatórias: 35-40, Amistoso: 20).
- Multiplicador adicional por diferença de gols (goal difference multiplier), penalizando/recompensando goleadas.
- Nomes históricos de seleções (ex: *Soviet Union*, *West Germany*, *Czechoslovakia*) são normalizados para
  o nome atual via `former_names.csv`, mantendo continuidade da série de rating.

**Modelo Poisson Bivariado**
- Força ofensiva e defensiva de cada seleção calculada com base nos últimos 10 anos de jogos (casa e fora).
- λ (gols esperados) = média histórica de gols × força ofensiva do time × força defensiva do adversário.
- Matriz de probabilidades de placar via distribuição de Poisson independente para cada seleção.
- Probabilidades de vitória/empate/derrota obtidas pela soma triangular da matriz.

**Limitações**
- O modelo não captura lesões, escalações, condições de jogo ou contexto situacional (jogo decisivo, dérbi etc.).
- Seleções com poucos jogos recentes (<5 no período de 10 anos) não entram no modelo Poisson.
- Indicado para fins analíticos e exploratórios, não como recomendação de apostas.
    """)

st.markdown(
    f'<div style="text-align:center; color:{TEXT_MUTED}; font-size:0.75rem; margin-top:2rem; '
    f'font-family:\'IBM Plex Mono\',monospace;">EDUMETRIA RESEARCH — Quantitative Football Analytics</div>',
    unsafe_allow_html=True
)
