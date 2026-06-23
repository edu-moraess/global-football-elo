"""
Global Football Quant Intelligence v3.1 — Arquitetura Modular
Entry point: apenas carrega dados globais e inicia navegação.
"""
import streamlit as st

st.set_page_config(
    page_title="Global Football Quant Intelligence",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; }
    h1, h2, h3, h4 { color: #f0f0f0; }
    .stMetric { background: #1a1a2e; border-radius: 8px; padding: 8px; }
    .stMarkdown { color: #ccc; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; margin-bottom:0;'>⚽ Global Football Quant Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888; font-size:1.1em; margin-top:4px;'>v3.1 — A Ciência por Trás da História</p>", unsafe_allow_html=True)
st.divider()

st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 24px; border-radius: 12px; margin-bottom: 20px;">
<p style="color: #ccc; line-height: 1.8; margin: 0;">
O futebol não é apenas 22 homens correndo atrás de uma bola. É matemática, história, psicologia e geografia 
condensados em 90 minutos. Este projeto usa o <b>motor Elo-Poisson-Dixon-Coles</b> para decifrar o jogo 
como nunca antes: não para apostar, mas para <b>entender</b>.
</p>
</div>
""", unsafe_allow_html=True)

st.info("""
👈 Use a **barra lateral** para navegar entre as seções analíticas:

- 🧠 **Inteligência Elo** — Ranking global com momentum, volatilidade e eficiência
- 🔮 **Predição Quant** — Modelo Poisson-Dixon-Coles + Monte Carlo para placares
- ⚔️ **Confronto Histórico** — H2H profundo com contexto psicológico e timeline
- 📊 **Métricas Avançadas** — Forma, perfil tático, clutch, resiliência e ciclos de geração
- 🏟️ **Contexto de Campo** — Home advantage por confederação e impacto da neutralidade
- 🧬 **Evolução Temporal** — Ciclos de geração, janelas de ouro e tendências de longo prazo
- 🏆 **Simulador de Torneios** — Simulação de grupos e mata-mata (Copa do Mundo)
- 🧮 **Laboratório Quant** — Análise de sensibilidade, cenários e robustez do modelo
""")

st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 30px;">
© 2026 Global Football Quant Intelligence | Motor Elo-Poisson-Dixon-Coles | Dados históricos internacionais<br>
<em>"O modelo não é perfeito. Mas é honesto."</em>
</div>
""", unsafe_allow_html=True)
