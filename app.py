import streamlit as st

st.set_page_config(
    page_title="Global Football Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.theme import inject_css, GOLD, MUTED, CARD, BORDER, TEXT, GREEN, BLUE, RED

inject_css()

st.markdown(
    f"""
    <div style='padding:28px 0 18px;border-bottom:1px solid {BORDER};margin-bottom:24px'>
        <div style='color:{MUTED};font-size:0.65rem;text-transform:uppercase;
                    letter-spacing:.18em;font-weight:600;margin-bottom:6px'>
            Quantitative Football Analytics
        </div>
        <h1 style='color:{GOLD};font-size:2.2rem;font-weight:700;
                   letter-spacing:-.03em;margin:0;line-height:1.1'>
            Global Football<br>Quant Intelligence
        </h1>
        <p style='color:{MUTED};font-size:0.85rem;margin-top:10px;max-width:540px;line-height:1.7'>
            Motor probabilístico sobre 49k+ partidas internacionais desde 1872.
            Elo com K-factor dinâmico · Dixon-Coles MLE · Monte Carlo 30k iterações.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

modules = [
    ("🧠", "Inteligência Elo",    "Ranking global, momentum, volatilidade e força de agenda por seleção."),
    ("🔮", "Predição Quant",      "Matriz de placares Dixon-Coles, mercados de gol, xG e Monte Carlo."),
    ("⚔️", "Confronto Histórico", "Head-to-head psicológico, dominância histórica e lollipop timeline."),
    ("📊", "Métricas Avançadas",  "Brier Score, clutch index, eficiência, curva de geração e xGOE."),
    ("🌍", "Contexto de Campo",   "Home advantage por confederação e análise de campo neutro."),
    ("📈", "Evolução Temporal",   "Série histórica de rating, ciclos de geração e janela de pico."),
    ("🏆", "Simulador de Torneios","Monte Carlo de grupos, mata-matas e Copa do Mundo completa."),
    ("🧪", "Laboratório Quant",   "Análise de sensibilidade, cenários what-if e Kelly Criterion."),
]

cols = st.columns(2)
for i, (icon, name, desc) in enumerate(modules):
    with cols[i % 2]:
        st.markdown(
            f"""<div style='background:{CARD};border:1px solid {BORDER};
                           border-left:2px solid {GOLD};border-radius:4px;
                           padding:14px 16px;margin-bottom:10px'>
                    <div style='display:flex;align-items:center;gap:10px;margin-bottom:5px'>
                        <span style='font-size:1.2rem'>{icon}</span>
                        <span style='color:{TEXT};font-size:0.88rem;font-weight:600'>{name}</span>
                    </div>
                    <div style='color:{MUTED};font-size:0.78rem;line-height:1.5'>{desc}</div>
                </div>""",
            unsafe_allow_html=True,
        )

st.markdown(
    f"<div style='margin-top:24px;padding-top:14px;border-top:1px solid {BORDER};"
    f"color:{MUTED};font-size:0.68rem;display:flex;justify-content:space-between'>"
    f"<span>© Global Football Quant Intelligence</span>"
    f"<span>Dados: International Football Results 1872–2025 · Transfermarkt</span>"
    f"</div>",
    unsafe_allow_html=True,
)