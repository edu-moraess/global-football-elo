"""
Global Football Intelligence
Painel quantitativo: Elo Ratings, Head-to-Head, Predição Poisson
Dataset: International football results (1872-2024) + Transfermarkt + Reep (técnicos) + Estádios
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

ACCENT = "#d4af37"    # gold
ACCENT2 = "#1f77b4"   # blue
POSITIVE = "#2ca02c"
NEGATIVE = "#d62728"

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
    "🗺️ Estádios & Técnicos"
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
        'λ (lambda) representa o número esperado de gols de cada seleção.</div>',
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
        elo_h = CURRENT_RATINGS.get(home_pred, INITIAL_ELO)
        elo_a = CURRENT_RATINGS.get(away_pred, INITIAL_ELO)
        p_elo_home, p_elo_away = predict_match(elo_h, elo_a, neutral=neutral_pred)

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

        m4, m5, m6 = st.columns(3)
        m4.metric(f"λ gols {home_pred}", f"{result['lambda_home']:.2f}")
        m5.metric(f"λ gols {away_pred}", f"{result['lambda_away']:.2f}")
        m6.metric("Elo (referência)", f"{elo_h:.0f} vs {elo_a:.0f}")

        st.caption(f"Probabilidade implícita por Elo: {home_pred} {p_elo_home*100:.1f}% — {away_pred} {p_elo_away*100:.1f}% (modelo simplificado)")

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

    torneios = sorted(RESULTS["tournament"].dropna().unique())
    selected_tournament = st.selectbox("Escolha uma competição", torneios)

    df_tourn = RESULTS[RESULTS["tournament"] == selected_tournament].copy()
    if df_tourn.empty:
        st.warning("Nenhuma partida encontrada.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        total_jogos = len(df_tourn)
        total_gols = df_tourn["home_score"].sum() + df_tourn["away_score"].sum()
        media_gols = total_gols / total_jogos
        times_distintos = pd.concat([df_tourn["home_team"], df_tourn["away_team"]]).nunique()

        col1.metric("Total de partidas", total_jogos)
        col2.metric("Total de gols", total_gols)
        col3.metric("Média de gols/jogo", f"{media_gols:.2f}")
        col4.metric("Países participantes", times_distintos)

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
            st.subheader("Artilheiros do torneio")
            st.dataframe(top_scorers, use_container_width=True, hide_index=True)

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

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jogadores na base", f"{len(PLAYERS):,}".replace(",", "."))
        m2.metric("Clubes", f"{len(CLUBS):,}".replace(",", "."))
        m3.metric("Transferências registradas", f"{len(TRANSFERS):,}".replace(",", "."))
        total_fees = TRANSFERS["transfer_fee"].sum()
        m4.metric("Volume total negociado", f"€ {total_fees/1e9:.1f} bi")

        st.markdown("---")
        # 1. Fluxo de transferências entre ligas
        st.markdown("### 🌍 Fluxo de Transferências entre Ligas")
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
                                         title=f"Fluxo de investimento – {season_sel}")
                st.plotly_chart(fig_sankey, use_container_width=True)

        st.markdown("---")
        # 2. Curva de valorização por idade
        st.markdown("### 📈 Curva de Valorização de Mercado por Idade")
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
                                  xaxis_title="Idade", yaxis_title="Valor de mercado (€ milhões)",
                                  title=f"Valor de Mercado por Idade — {position_sel}")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        # 3. Maiores transferências + nacionalidades
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("### 💸 Maiores Transferências da História")
            tt = top_transfers(TRANSFERS, n=15, min_fee=1_000_000)
            tt_display = tt.copy()
            tt_display["transfer_fee"] = (tt_display["transfer_fee"] / 1e6).round(1).astype(str) + " M€"
            tt_display.columns = ["Data", "Jogador", "De", "Para", "Valor", "Temporada"]
            st.dataframe(tt_display, use_container_width=True, hide_index=True, height=400)

        with col6:
            st.markdown("### 🌎 Nacionalidade dos Jogadores por Liga")
            comp_options = ["Todas"] + sorted(PLAYERS["current_club_domestic_competition_id"].dropna().unique().tolist())
            comp_labels = {c: COMPETITION_LABELS.get(c, c) for c in comp_options}
            comp_sel = st.selectbox("Liga", comp_options, format_func=lambda x: comp_labels.get(x, x))
            nat = nationality_distribution(PLAYERS, comp_sel).head(12)
            fig = px.bar(nat, x="Jogadores", y="País", orientation="h",
                         title=f"Top Nacionalidades — {comp_labels.get(comp_sel, comp_sel)}")
            fig.update_traces(marker_color=ACCENT)
            fig.update_layout(template=PLOTLY_TEMPLATE, height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        # 4. Explorador de clubes
        st.markdown("### 🏟️ Explorador de Clubes")
        league_options = ["Todas"] + sorted(CLUBS["domestic_competition_id"].dropna().unique().tolist())
        league_labels = {l: COMPETITION_LABELS.get(l, l) for l in league_options}
        league_sel = st.selectbox("Filtrar por liga", league_options, format_func=lambda x: league_labels.get(x, x), key="club_league")
        cs = club_summary(CLUBS, PLAYERS, league_sel).sort_values("squad_value_eur", ascending=False).head(20)
        cs_display = cs.copy()
        cs_display["squad_value_eur"] = (cs_display["squad_value_eur"] / 1e6).round(1).astype(str) + " M€"
        cs_display.columns = ["Clube", "Elenco", "Idade Média", "% Estrangeiros", "Jogadores Sel. Nacional",
                              "Valor do Elenco", "Estádio", "Capacidade"]
        st.dataframe(cs_display, use_container_width=True, hide_index=True, height=400)

        st.markdown("---")
        # 5. Conexão Seleção ↔ Clube
        st.markdown("### 🔗 Conexão Seleção ↔ Clube")
        st.caption("Jogadores com mais presenças/gols pela seleção e o valor de mercado atual em seus clubes.")
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
            bridge_display.columns = ["Jogador", "Clube Atual", "Posição", "Jogos pela Seleção", "Gols pela Seleção", "Valor de Mercado"]
            st.dataframe(bridge_display, use_container_width=True, hide_index=True)


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

    ### 🗺️ **Estádios e Técnicos**
    - Estádios com coordenadas geográficas para visualização em mapa.
    - Dados de técnicos provenientes do **Reep (people.csv)** – mais de 700 mil registros, filtrados para `type == 'coach'`.

    ### 📂 **Fontes**
    - Partidas internacionais: [Football Results (1872-2024)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
    - Dados de mercado: [Transfermarkt Dataset](https://www.kaggle.com/datasets/davidcariboo/player-scores)
    - Técnicos: [Reep - Football Entity Register](https://github.com/withqwerty/reep)
    - Estádios: WorldSoccerStadiums + Football Stadiums (Kaggle)
    - Processamento e visualização: **Python, Pandas, Plotly, Streamlit**.

    ---
    **Desenvolvido por Eduardo Moraes**  
    [GitHub](https://github.com/edu-moraess/global-football-elo)
    """)


# ---------------------------------------------------------------------------
# TAB 7 — ESTÁDIOS & TÉCNICOS
# ---------------------------------------------------------------------------
with tab7:
    st.subheader("🗺️ Mapa Mundial de Estádios")

    # ------------------------------
    # 1. Carregar dados de estádios
    # ------------------------------
    @st.cache_data
    def load_stadiums_data():
        gps_path = Path("data/futebol_stadiums.csv")
        info_path = Path("data/Football Stadiums.csv")

        stadiums_gps = pd.DataFrame()
        stadiums_info = pd.DataFrame()

        if gps_path.exists():
            try:
                stadiums_gps = pd.read_csv(gps_path, encoding='utf-8')
            except UnicodeDecodeError:
                stadiums_gps = pd.read_csv(gps_path, encoding='latin1')
            stadiums_gps["Latitude"] = pd.to_numeric(stadiums_gps.get("Latitude", pd.Series()), errors="coerce")
            stadiums_gps["Longitude"] = pd.to_numeric(stadiums_gps.get("Longitude", pd.Series()), errors="coerce")
            stadiums_gps = stadiums_gps.dropna(subset=["Latitude", "Longitude"])
        else:
            st.warning("Arquivo 'futebol_stadiums.csv' (com coordenadas) não encontrado em data/")

        if info_path.exists():
            try:
                stadiums_info = pd.read_csv(info_path, encoding='utf-8')
            except UnicodeDecodeError:
                stadiums_info = pd.read_csv(info_path, encoding='latin1')
            stadiums_info.rename(columns={
                "Stadium": "Stadium",
                "Capacity": "Capacity",
                "Country": "Country",
                "Confederation": "Confederation",
                "City": "City",
                "HomeTeams": "Team"
            }, inplace=True)

        if not stadiums_gps.empty and not stadiums_info.empty:
            stadiums_info["Stadium_clean"] = stadiums_info["Stadium"].str.strip().str.lower()
            stadiums_gps["Stadium_clean"] = stadiums_gps["Stadium"].str.strip().str.lower()
            stadiums_gps = stadiums_gps.merge(
                stadiums_info[["Stadium_clean", "Capacity", "Confederation", "Team"]],
                on="Stadium_clean",
                how="left",
                suffixes=("", "_info")
            )
            if "Capacity_info" in stadiums_gps.columns:
                stadiums_gps["Capacity"] = stadiums_gps["Capacity_info"].fillna(stadiums_gps.get("Capacity", pd.Series()))
                stadiums_gps.drop(columns=["Capacity_info", "Stadium_clean"], inplace=True)
            else:
                stadiums_gps.drop(columns=["Stadium_clean"], inplace=True)

        return stadiums_gps

    df_stadiums = load_stadiums_data()

    if not df_stadiums.empty:
        st.map(df_stadiums, latitude="Latitude", longitude="Longitude", size=10, zoom=1)

        st.markdown("---")
        st.subheader("📋 Explorador de Estádios")

        col1, col2 = st.columns(2)
        with col1:
            paises = sorted(df_stadiums["Country"].dropna().unique())
            pais_sel = st.selectbox("Filtrar por país", ["Todos"] + paises)
        with col2:
            if "Team" in df_stadiums.columns:
                times = sorted(df_stadiums["Team"].dropna().unique())
                time_sel = st.selectbox("Filtrar por time mandante", ["Todos"] + times)
            else:
                time_sel = "Todos"

        df_filtrado = df_stadiums.copy()
        if pais_sel != "Todos":
            df_filtrado = df_filtrado[df_filtrado["Country"] == pais_sel]
        if time_sel != "Todos" and "Team" in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado["Team"] == time_sel]

        colunas_exibir = ["Stadium", "City", "Country", "Capacity", "Team"]
        colunas_disponiveis = [c for c in colunas_exibir if c in df_filtrado.columns]
        st.dataframe(
            df_filtrado[colunas_disponiveis],
            use_container_width=True,
            hide_index=True,
            height=400
        )
        st.caption(f"Mostrando {len(df_filtrado)} estádios de um total de {len(df_stadiums)} com coordenadas.")
    else:
        st.info("Nenhum dado de estádio com coordenadas disponível. Adicione o arquivo 'futebol_stadiums.csv' na pasta data/")

    # -----------------------------------------------------------------------
    # 2. Dados de Técnicos (Reep - suporte a N partes people_parte*.csv)
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🧑‍🏫 Dados de Técnicos (Reep)")

    @st.cache_data
    def load_coaches_data():
        data_dir = Path("data")
        parte_files = sorted(data_dir.glob("people_parte*.csv"))
        original_file = data_dir / "people.csv"

        if parte_files:
            dfs = []
            for file in parte_files:
                try:
                    df_part = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    df_part = pd.read_csv(file, encoding='latin1')
                dfs.append(df_part)
                st.info(f"📂 Carregado: {file.name} ({len(df_part):,} linhas)")
            df_full = pd.concat(dfs, ignore_index=True)
            st.success(f"✅ Total carregado das {len(parte_files)} partes: {len(df_full):,} registros.")
        elif original_file.exists():
            try:
                df_full = pd.read_csv(original_file, sep=';', encoding='utf-8')
            except UnicodeDecodeError:
                df_full = pd.read_csv(original_file, sep=';', encoding='latin1')
            st.success(f"✅ Carregado arquivo original: {len(df_full):,} registros.")
        else:
            st.warning("Nenhum arquivo de técnicos encontrado (people_parte*.csv ou people.csv).")
            return pd.DataFrame()

        # Limpa espaços em branco dos nomes das colunas
        df_full.columns = df_full.columns.str.strip()

        # Procura coluna 'type' de forma case-insensitive
        type_col = None
        for col in df_full.columns:
            if col.lower() == 'type':
                type_col = col
                break

        if type_col:
            df_coaches = df_full[df_full[type_col] == 'coach'].copy()
            st.caption(f"🎯 Filtrados {len(df_coaches):,} técnicos (type='coach').")
        else:
            df_coaches = df_full.copy()
            st.caption("Coluna 'type' não encontrada. Exibindo todos os registros.")

        return df_coaches

    df_coaches = load_coaches_data()

    if not df_coaches.empty:
        colunas_coach = df_coaches.columns.tolist()
        st.caption(f"📋 Colunas disponíveis: {', '.join(colunas_coach[:8])} ...")

        # Gráfico de nacionalidades
        if 'nationality' in colunas_coach:
            top_nats = df_coaches['nationality'].value_counts().head(10).reset_index()
            top_nats.columns = ['Nacionalidade', 'Quantidade']
            fig = px.bar(top_nats, x='Quantidade', y='Nacionalidade', orientation='h',
                         title='Técnicos por nacionalidade', color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, use_container_width=True)
        elif 'country_of_birth' in colunas_coach:
            top_nats = df_coaches['country_of_birth'].value_counts().head(10).reset_index()
            top_nats.columns = ['País', 'Quantidade']
            fig = px.bar(top_nats, x='Quantidade', y='País', orientation='h',
                         title='Técnicos por país de nascimento', color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, use_container_width=True)
        elif 'country' in colunas_coach:
            top_nats = df_coaches['country'].value_counts().head(10).reset_index()
            top_nats.columns = ['País', 'Quantidade']
            fig = px.bar(top_nats, x='Quantidade', y='País', orientation='h',
                         title='Técnicos por país', color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, use_container_width=True)

        # Filtro por país
        coluna_pais = None
        if 'country' in colunas_coach:
            coluna_pais = 'country'
        elif 'country_of_birth' in colunas_coach:
            coluna_pais = 'country_of_birth'
        elif 'nationality' in colunas_coach:
            coluna_pais = 'nationality'

        if coluna_pais:
            paises_coach = sorted(df_coaches[coluna_pais].dropna().unique())
            pais_coach_sel = st.selectbox("Filtrar técnicos por país", ["Todos"] + paises_coach)
            if pais_coach_sel != "Todos":
                df_coaches = df_coaches[df_coaches[coluna_pais] == pais_coach_sel]

        # Exibir tabela
        colunas_exibir_coach = []
        for col in ['name', 'full_name', 'date_of_birth', 'age', 'nationality', 'country', 'country_of_birth', 'position']:
            if col in df_coaches.columns:
                colunas_exibir_coach.append(col)
        if not colunas_exibir_coach:
            colunas_exibir_coach = df_coaches.columns[:5]

        st.markdown("**📋 Tabela de técnicos (primeiras 100 linhas)**")
        st.dataframe(df_coaches[colunas_exibir_coach].head(100), use_container_width=True, hide_index=True)
    else:
        st.info("📂 Nenhum dado de técnicos carregado. Coloque os arquivos na pasta data/")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    f'<div style="text-align:center; font-size:0.75rem; opacity:0.6; margin-top:2rem; '
    f'font-family:\'IBM Plex Mono\',monospace;">Global Football Intelligence — Quant Analytics</div>',
    unsafe_allow_html=True
)