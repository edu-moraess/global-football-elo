"""
Global Football Intelligence
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

# Accent colors that work on both light and dark Streamlit themes
ACCENT = "#d4af37"    # gold
ACCENT2 = "#1f77b4"   # blue
POSITIVE = "#2ca02c"
NEGATIVE = "#d62728"

# Plotly template that adapts to Streamlit's active theme (no hardcoded bg)
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", size=12),
        colorway=[ACCENT, ACCENT2, "#f72585", "#4ade80", "#fb923c", "#a78bfa"],
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)

st.markdown(f"""
<style>
h1, h2, h3 {{ font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.5px; }}
h1 {{ color: {ACCENT} !important; border-bottom: 2px solid {ACCENT}; padding-bottom: 12px; }}
[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem; }}
.stTabs [aria-selected="true"] {{ color: {ACCENT} !important; border-bottom-color: {ACCENT} !important; }}
.block-container {{ padding-top: 1.5rem; }}
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
    """Build last-N form pills (W/D/L) for a team, most recent first."""
    sub = results_df[(results_df["home_team"] == team) | (results_df["away_team"] == team)].sort_values("date", ascending=False).head(n)
    html = ""
    for _, r in sub.iterrows():
        is_home = r["home_team"] == team
        gf = r["home_score"] if is_home else r["away_score"]
        ga = r["away_score"] if is_home else r["home_score"]
        if gf > ga:
            cls, letter = "form-w", "V"
        elif gf < ga:
            cls, letter = "form-l", "D"
        else:
            cls, letter = "form-d", "E"
        html += f'<span class="form-pill {cls}">{letter}</span>'
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
        st.warning("Diretório 'data_clubs' não encontrado ou arquivos ausentes. Módulo de clubes indisponível.")
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Elo Ranking & Evolução", "🆚 Head-to-Head", "🎯 Predição Poisson",
    "🏆 Tournament Analytics", "💰 Player Market Intelligence", "ℹ️ Metodologia"
])


# ---------------------------------------------------------------------------
# TAB 1 — ELO RANKING
# ---------------------------------------------------------------------------
with tab1:
    col_a, col_b = st.columns([1, 2])

    ranking_df = pd.DataFrame(
        [{"Seleção": k, "Elo": round(v, 1), "Tier": elo_tier(v)[0]} for k, v in CURRENT_RATINGS.items()]
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
            column_config={
                "Elo": st.column_config.ProgressColumn(
                    "Elo", min_value=1300, max_value=2100, format="%.0f"
                ),
            },
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

        elo_a_val = CURRENT_RATINGS.get(team_a, INITIAL_ELO)
        elo_b_val = CURRENT_RATINGS.get(team_b, INITIAL_ELO)
        tier_a, color_a = elo_tier(elo_a_val)
        tier_b, color_b = elo_tier(elo_b_val)

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

        # Pizza de resultados + linha temporal de gols
        col1, col2 = st.columns([1, 2])
        with col1:
            pie = go.Figure(data=[go.Pie(
                labels=[f"{team_a}", "Empate", f"{team_b}"],
                values=[wins_a, draws, wins_b],
                hole=0.5,
                marker=dict(colors=[ACCENT, "#999999", ACCENT2]),
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
            colors = [ACCENT if v > 0 else (ACCENT2 if v < 0 else "#999999") for v in h2h_sorted["goal_diff"]]
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

        # Stacked probability gauge
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
            colorscale="YlOrRd",
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
# TAB 4 — TOURNAMENT ANALYTICS
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Análise por Competição")

    # Obter lista de torneios (excluindo valores nulos)
    torneios = sorted(RESULTS["tournament"].dropna().unique())
    selected_tournament = st.selectbox("Escolha uma competição", torneios)

    # Filtrar partidas
    df_tourn = RESULTS[RESULTS["tournament"] == selected_tournament].copy()
    if df_tourn.empty:
        st.warning("Nenhuma partida encontrada para esta competição.")
    else:
        # Métricas gerais
        col1, col2, col3, col4 = st.columns(4)
        total_jogos = len(df_tourn)
        total_gols = df_tourn["home_score"].sum() + df_tourn["away_score"].sum()
        media_gols = total_gols / total_jogos
        times_distintos = pd.concat([df_tourn["home_team"], df_tourn["away_team"]]).nunique()

        col1.metric("Total de partidas", total_jogos)
        col2.metric("Total de gols", total_gols)
        col3.metric("Média de gols/jogo", f"{media_gols:.2f}")
        col4.metric("Países participantes", times_distintos)

        # Artilheiros (se houver dados)
        gs = DATA["goalscorers"]
        gs_tourn = gs.merge(
            df_tourn[["date", "home_team", "away_team"]],
            on=["date", "home_team", "away_team"],
            how="inner"
        )
        gs_tourn = gs_tourn[gs_tourn["own_goal"] != True]  # excluir gols contra
        if not gs_tourn.empty:
            top_scorers = (
                gs_tourn.groupby("scorer")
                .size()
                .reset_index(name="gols")
                .sort_values("gols", ascending=False)
                .head(10)
            )
            st.subheader("Artilheiros do torneio")
            st.dataframe(top_scorers, use_container_width=True, hide_index=True)
        else:
            st.info("Dados de artilheiros não disponíveis para esta competição.")

        # Evolução dos gols ao longo das edições (agrupando por ano)
        df_tourn["year"] = df_tourn["date"].dt.year
        gols_por_ano = df_tourn.groupby("year").apply(
            lambda x: x["home_score"].sum() + x["away_score"].sum()
        ).reset_index(name="gols")
        fig = px.line(gols_por_ano, x="year", y="gols", title="Evolução de gols por ano")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 5 — PLAYER MARKET INTELLIGENCE
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Inteligência de Mercado (Transfermarkt)")

    # Verificar se dados de clubes foram carregados
    if CLUB_DATA is None:
        st.error("Dados de clubes não disponíveis. Verifique os arquivos em data_clubs/")
    else:
        # Sub-abas internas
        sub_tab5 = st.tabs([
            "📈 Curva de Valorização por Idade",
            "💸 Maiores Transferências",
            "🌍 Fluxo de Transferências entre Ligas",
            "🧬 Distribuição de Nacionalidades",
            "🏢 Resumo por Clube"
        ])

        # 1. Curva de valorização por idade
        with sub_tab5[0]:
            st.markdown("**Valor médio e mediano de mercado por idade**")
            # Obter posições disponíveis
            positions = ["Todas"] + sorted(CLUB_DATA["players"]["position"].dropna().unique())
            pos_filter = st.selectbox("Filtrar por posição", positions, key="pos_curve")
            curve_df = valuation_age_curve(
                CLUB_DATA["valuations"],
                CLUB_DATA["players"],
                position_filter=None if pos_filter == "Todas" else pos_filter,
                min_value=10000
            )
            if curve_df is not None and not curve_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=curve_df["age_bucket"], y=curve_df["mean"],
                    mode="lines+markers", name="Média", line=dict(color=ACCENT)
                ))
                fig.add_trace(go.Scatter(
                    x=curve_df["age_bucket"], y=curve_df["median"],
                    mode="lines+markers", name="Mediana", line=dict(color=ACCENT2, dash="dash")
                ))
                fig.update_layout(
                    template=PLOTLY_TEMPLATE,
                    title=f"Valor de mercado por idade – {pos_filter}",
                    xaxis_title="Idade (anos)",
                    yaxis_title="Valor de mercado (€)",
                    hovermode="x"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Dados insuficientes para esta posição/filtro.")

        # 2. Maiores transferências
        with sub_tab5[1]:
            st.markdown("**Top 20 transferências mais caras**")
            top_df = top_transfers(CLUB_DATA["transfers"], n=20, min_fee=1_000_000)
            # Formatar valores
            top_df["transfer_fee_meur"] = top_df["transfer_fee"] / 1e6
            top_df = top_df.rename(columns={
                "transfer_date": "Data",
                "player_name": "Jogador",
                "from_club_name": "Clube Origem",
                "to_club_name": "Clube Destino",
                "transfer_fee_meur": "Valor (M€)",
                "transfer_season": "Temporada"
            })
            st.dataframe(top_df[["Data", "Jogador", "Clube Origem", "Clube Destino", "Valor (M€)", "Temporada"]],
                         use_container_width=True, hide_index=True)

        # 3. Fluxo de transferências entre ligas (Sankey)
        with sub_tab5[2]:
            st.markdown("**Fluxo financeiro de transferências entre ligas**")
            # Permitir selecionar temporada
            seasons = sorted(CLUB_DATA["transfers"]["transfer_season"].dropna().unique())
            selected_season = st.selectbox("Temporada", ["Todas"] + list(seasons))
            season_param = None if selected_season == "Todas" else selected_season
            flow_df = transfer_flow_by_league(
                CLUB_DATA["transfers"],
                CLUB_DATA["clubs"],
                season=season_param,
                top_n=12,
                min_fee=500_000
            )
            if not flow_df.empty:
                # Preparar nós e links para Sankey
                labels = list(pd.unique(flow_df[["from_league", "to_league"]].values.ravel()))
                label_to_idx = {label: i for i, label in enumerate(labels)}
                sources = flow_df["from_league"].map(label_to_idx).tolist()
                targets = flow_df["to_league"].map(label_to_idx).tolist()
                values = flow_df["transfer_fee"].tolist()

                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=labels,
                        color=ACCENT
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color="rgba(31, 119, 180, 0.4)"
                    )
                )])
                fig_sankey.update_layout(title="Fluxo de investimento entre ligas (€)", height=600)
                st.plotly_chart(fig_sankey, use_container_width=True)
            else:
                st.info("Nenhum fluxo significativo para os filtros selecionados.")

        # 4. Distribuição de nacionalidades
        with sub_tab5[3]:
            st.markdown("**Nacionalidades mais representadas**")
            # Permitir filtrar por competição
            comps = ["Todas"] + sorted(CLUB_DATA["players"]["current_club_domestic_competition_id"].dropna().unique())
            comp_filter = st.selectbox("Liga", comps)
            comp_id = None if comp_filter == "Todas" else comp_filter
            nat_df = nationality_distribution(CLUB_DATA["players"], competition_id=comp_id)
            if not nat_df.empty:
                top_nat = nat_df.head(15)
                fig_nat = px.bar(top_nat, x="Jogadores", y="País", orientation="h",
                                 title="Número de jogadores por nacionalidade")
                st.plotly_chart(fig_nat, use_container_width=True)
            else:
                st.info("Dados insuficientes.")

        # 5. Resumo por clube
        with sub_tab5[4]:
            st.markdown("**Visão geral de clubes**")
            club_summary_df = club_summary(CLUB_DATA["clubs"], CLUB_DATA["players"])
            if club_summary_df is not None and not club_summary_df.empty:
                club_names = sorted(club_summary_df["name"].unique())
                selected_club = st.selectbox("Selecione um clube", club_names)
                club_row = club_summary_df[club_summary_df["name"] == selected_club].iloc[0]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Valor do elenco", f"€{club_row['squad_value_eur']/1e9:.2f}B")
                col2.metric("Elenco", int(club_row['squad_size']))
                col3.metric("Idade média", f"{club_row['average_age']:.1f}")
                col4.metric("Estrangeiros", f"{club_row['foreigners_percentage']:.1f}%")
                st.metric("Jogadores na seleção nacional", int(club_row['national_team_players']))
            else:
                st.warning("Não foi possível carregar dados dos clubes.")


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

    ### 📂 **Fontes**
    - Partidas internacionais: [Football Results (1872-2024)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
    - Dados de mercado: [Transfermarkt Dataset](https://www.kaggle.com/datasets/davidcariboo/player-scores)
    - Processamento e visualização: **Python, Pandas, Plotly, Streamlit**.

    ---
    **Desenvolvido por Eduardo Moraes**  
    [GitHub](https://github.com/edu-moraess/global-football-elo)
    """)